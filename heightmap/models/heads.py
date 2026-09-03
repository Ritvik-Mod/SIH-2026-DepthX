"""Replacement heads for the Depth Anything decoder.

Stock DAv2 emits affine-invariant inverse depth: its training loss was built so that
a*d+b scores identically to d, which is why absolute scale cannot be recovered by
post-hoc rescaling and the head has to be replaced rather than calibrated.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise linear modulation on log(GSD).

    A 40-pixel-wide flat rectangle is a shed at 0.3 m/px and a warehouse at 2 m/px:
    identical pixels, different object, different plausible height.  This tells the
    decoder how to interpret the scale of what the encoder saw.  log because the
    resolutions of interest span orders of magnitude.
    """

    def __init__(self, channels: int, hidden: int = 64, gsd_ref: float = 0.5):
        super().__init__()
        self.gsd_ref = gsd_ref
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)      # starts as identity: f -> f*(1+0)+0

    def forward(self, f: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        z = torch.log(gsd.clamp(min=1e-3).reshape(-1, 1) / self.gsd_ref)
        gamma, beta = self.mlp(z.to(f.dtype)).chunk(2, dim=1)
        return f * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


def _stem(in_ch: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, padding=1),
        nn.GroupNorm(min(8, hidden), hidden),
        nn.GELU(),
        nn.Conv2d(hidden, hidden, 3, padding=1),
        nn.GELU(),
    )


class RegressionHeightHead(nn.Module):
    """Direct metric regression.  Simple, and the honest floor for the ablation table."""

    def __init__(self, in_ch: int, h_max: float, hidden: int = 64, init_height: float = 8.0):
        super().__init__()
        self.h_max = h_max
        self.stem = _stem(in_ch, hidden)
        self.out = nn.Conv2d(hidden, 1, 1)
        # sigmoid(0) = 0.5 puts the initial prediction at h_max/2 -- about 125 m for a
        # scene whose median height is 4 m.  Bias the output so it starts at
        # init_height instead: b = logit(init_height / h_max).
        with torch.no_grad():
            self.out.weight.zero_()
            f = min(max(init_height / max(h_max, 1e-6), 1e-4), 1 - 1e-4)
            self.out.bias.fill_(float(torch.logit(torch.tensor(f))))

    def forward(self, f, out_hw):
        h = self.h_max * torch.sigmoid(self.out(self.stem(f)))
        return {"height": F.interpolate(h, out_hw, mode="bilinear", align_corners=False),
                "height_dec": h}


class BinsHeightHead(nn.Module):
    """Adaptive-bins classification-regression (AdaBins family).

    A pixel at a roof edge is either ground or roof, never the average of the two.
    Direct regression is forced to emit that average -- a value in mid-air -- which is
    what produces soft, ramped building edges.  A distribution over bins can stay
    bimodal, and the expectation still gives a continuous sub-bin value.

    Bin edges are predicted per image so a scene of bungalows and a scene of towers
    each get bins spanning their own range instead of sharing one fixed global grid.
    """

    def __init__(self, in_ch: int, n_bins: int, h_max: float, hidden: int = 64,
                 min_width: float = 1e-3, init_height: float = 8.0):
        super().__init__()
        self.n_bins, self.h_max, self.min_width = n_bins, h_max, min_width
        self.stem = _stem(in_ch, hidden)
        self.logits = nn.Conv2d(hidden, n_bins, 1)
        self.bin_mlp = nn.Sequential(nn.Linear(in_ch, hidden), nn.GELU(), nn.Linear(hidden, n_bins))
        # Uniform bin probabilities put the initial expectation at h_max/2 -- about
        # 125 m for a scene whose median height is 4 m.  That 100 m+ starting error
        # wastes the early steps and swamps the L1 term.
        #
        # Set the initial logits to an exponential decay exp(-i/tau) instead.  With bin
        # width w = h_max/N the expected index of that distribution is ~tau, so the
        # expected height is ~(tau+0.5)*w; solving for a target init_height gives
        # tau = init_height*N/h_max.  This adapts automatically to n_bins and h_max
        # rather than being a tuned magic ramp.
        with torch.no_grad():
            self.logits.weight.zero_()
            tau = max(init_height * n_bins / max(h_max, 1e-6), 0.5)
            self.logits.bias.copy_(-torch.arange(n_bins, dtype=torch.float32) / tau)

    def forward(self, f, out_hw):
        pooled = f.mean(dim=(2, 3))                              # (B,C)
        w = torch.softmax(self.bin_mlp(pooled), dim=1)           # (B,N) sums to 1
        w = w + self.min_width
        w = w / w.sum(dim=1, keepdim=True) * self.h_max          # (B,N) widths in metres
        edges = torch.cumsum(w, dim=1)                           # right edges
        centres = edges - w / 2                                  # (B,N)

        logits = self.logits(self.stem(f))                       # (B,N,h,w) at decoder res
        p = torch.softmax(logits, dim=1)
        h = (p * centres[:, :, None, None]).sum(dim=1, keepdim=True)
        return {"height": F.interpolate(h, out_hw, mode="bilinear", align_corners=False),
                "height_dec": h, "bin_logits": logits, "bin_centres": centres}


class SemanticHead(nn.Module):
    """Auxiliary land-cover head.  Regularises the shared features (height and class are
    strongly correlated) and produces the building mask needed for building-only metrics."""

    def __init__(self, in_ch: int, n_classes: int, hidden: int = 64):
        super().__init__()
        self.stem = _stem(in_ch, hidden)
        self.out = nn.Conv2d(hidden, n_classes, 1)

    def forward(self, f, out_hw):
        return F.interpolate(self.out(self.stem(f)), out_hw, mode="bilinear", align_corners=False)


class UncertaintyHead(nn.Module):
    """Predicts log sigma^2.  Log so exp() keeps the variance positive and the
    optimisation stable; a raw sigma head drifts negative and produces NaNs."""

    def __init__(self, in_ch: int, hidden: int = 64, clamp=(-8.0, 8.0)):
        super().__init__()
        self.clamp = clamp
        self.stem = _stem(in_ch, hidden)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, f, out_hw):
        lv = self.out(self.stem(f)).clamp(*self.clamp)
        return F.interpolate(lv, out_hw, mode="bilinear", align_corners=False)
