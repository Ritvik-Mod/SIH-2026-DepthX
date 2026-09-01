"""Individual loss terms.  Shapes throughout:
    pred, gt, mask : (B,1,H,W)   metres / metres / bool
    n = mask.sum()
Each returns a scalar.  The division of labour:
    SILog    structure, blind to global scale -> stable early training
    L1       absolute metres -> the only term that anchors scale
    grad     spatial derivative -> sharp walls instead of ramps
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

EPS = 1e-6


def _masked_mean(x, mask):
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def l1_loss(pred, gt, mask):
    """L1 not MSE: LiDAR-derived truth carries outliers (cranes, edge artefacts) and
    squaring lets a single bad pixel dominate the gradient."""
    return _masked_mean((pred - gt).abs(), mask)


def silog_loss(pred, gt, mask, lam: float = 0.85, alpha: float = 10.0, shift: float = 1.0):
    """Scale-invariant log error.

    g = log(d+1) - log(d*+1).  The +1 shift exists because AGL ground is exactly 0
    and log(0) is undefined -- depth data never has this problem, height data always does.

    mean(g^2) is total log error; mean(g)^2 is the systematic offset squared.
    Multiplying every height by c adds log(c) to every g, moving the mean but not the
    variance, so subtracting lam*mean(g)^2 removes the part of the error that is a
    uniform rescaling.  lam<1 leaves a nudge toward correct absolute scale.
    """
    g = torch.log(pred.clamp(min=0) + shift) - torch.log(gt.clamp(min=0) + shift)
    m = mask.to(g.dtype)
    n = m.sum().clamp(min=1.0)
    g = g * m
    mg2 = (g ** 2).sum() / n
    mg = g.sum() / n
    return alpha * torch.sqrt((mg2 - lam * mg ** 2).clamp(min=1e-12))


def multiscale_gradient_loss(pred, gt, mask, scales: int = 4, shift: float = 1.0):
    """L1 on the spatial derivative of the log residual, at several scales.

    L1 and SILog are pointwise: a prediction that ramps smoothly across a wall can
    score the same as one with a sharp step, because the errors cancel across the ramp.
    This term sees the difference.  L1 (not L2) on gradients encourages sparse gradient
    fields, i.e. piecewise-flat regions with sharp jumps -- which is what a city is.
    """
    g0 = (torch.log(pred.clamp(min=0) + shift) - torch.log(gt.clamp(min=0) + shift))
    m0 = mask.to(g0.dtype)
    g, m = g0 * m0, m0
    total = g0.new_zeros(())
    for k in range(scales):
        if k > 0:
            if min(g.shape[-2:]) < 4:
                break
            g = F.avg_pool2d(g, 2)
            m = F.avg_pool2d(m, 2)
        gk = torch.where(m > 0, g / m.clamp(min=EPS), torch.zeros_like(g))
        mk = (m > 0.5).to(gk.dtype)
        dx = gk[..., :, 1:] - gk[..., :, :-1]
        wx = mk[..., :, 1:] * mk[..., :, :-1]
        dy = gk[..., 1:, :] - gk[..., :-1, :]
        wy = mk[..., 1:, :] * mk[..., :-1, :]
        total = total + (dx.abs() * wx).sum() / wx.sum().clamp(min=1.0) \
                      + (dy.abs() * wy).sum() / wy.sum().clamp(min=1.0)
    return total / max(scales, 1)


def bin_ce_loss(bin_logits, bin_centres, gt, mask):
    """Cross-entropy of the per-pixel bin distribution against the bin nearest the truth.
    Computed at decoder resolution -- doing it at full res costs memory for no gain."""
    B, N, h, w = bin_logits.shape
    # bilinear rather than area: MPS cannot do adaptive pooling for non-divisible
    # sizes (518 -> 296), and this runs on the dev machine as well as the GPU box.
    gt_d = F.interpolate(gt, (h, w), mode="bilinear", align_corners=False)
    m_d = F.interpolate(mask.float(), (h, w), mode="bilinear", align_corners=False) > 0.99
    d = (gt_d.reshape(B, 1, h, w) - bin_centres.reshape(B, N, 1, 1)).abs()
    target = d.argmin(dim=1)                                   # (B,h,w)
    ce = F.cross_entropy(bin_logits, target, reduction="none").unsqueeze(1)
    return _masked_mean(ce, m_d)


def chamfer_bins_loss(bin_centres, gt, mask, samples: int = 4096):
    """Bidirectional nearest-neighbour distance between predicted bin centres and the
    heights actually present in the image.  Forward term: every real height should have
    a bin near it.  Backward term: every bin should sit near some real height, which
    stops the model parking unused bins in empty parts of the range."""
    B, N = bin_centres.shape
    total = bin_centres.new_zeros(())
    cnt = 0
    for b in range(B):
        v = gt[b][mask[b]]
        if v.numel() == 0:
            continue
        if v.numel() > samples:
            v = v[torch.randint(0, v.numel(), (samples,), device=v.device)]
        d = (v.reshape(-1, 1) - bin_centres[b].reshape(1, -1)).abs()
        total = total + d.min(dim=1).values.mean() + d.min(dim=0).values.mean()
        cnt += 1
    return total / max(cnt, 1)


def semantic_loss(logits, target, ignore_index: int = -1):
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


def gaussian_nll_loss(pred, gt, log_var, mask):
    """0.5 * [ (d-d*)^2 / sigma^2 + log sigma^2 ].
    First term says be accurate; second stops the model inflating sigma to escape it.
    Introduce only after the model already trains with L1 -- from scratch it discovers
    it can drive the loss down by making sigma enormous and stops learning accuracy."""
    inv = torch.exp(-log_var)
    return _masked_mean(0.5 * ((pred - gt) ** 2 * inv + log_var), mask)
