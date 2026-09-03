"""Full-scene inference.

Naive tiling produces a visible grid of height steps, and the reason points at the fix:
the model predicts height above LOCAL ground, so it has to find the ground.  A tile that
is entirely rooftop contains no ground at all, so the model guesses, and neighbouring
tiles guess differently.  The drift concentrates in dense urban tiles -- the ones that
matter.

Coarse-guided levelling gives every tile access to global context by borrowing it from
a whole-scene pass:
    1. predict on the downscaled whole scene   -> globally consistent, blurry
    2. predict per tile at full resolution     -> sharp, locally normalised
    3. level each tile onto the coarse reference (offset only, median)
    4. cosine-window blend the overlaps

Offset only, not a full affine: the model is GSD-conditioned and metric, so the scale is
already right, and fitting scale as well would let blur from the coarse pass leak into
the sharp tiles.  Median, not mean, so a few wild pixels cannot drag a whole tile.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from .data.transforms import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class InferResult:
    agl: np.ndarray                     # (H,W) float32 metres above local ground
    sigma: np.ndarray | None = None     # (H,W) float32 metres, if the model has the head
    overlap_disagreement: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def _normalise(img_u8: np.ndarray) -> np.ndarray:
    x = img_u8.astype(np.float32) / 255.0 if img_u8.dtype == np.uint8 else img_u8.astype(np.float32)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def _blend_window(tile: int, ramp: int) -> np.ndarray:
    w = np.ones(tile, np.float32)
    if ramp > 0:
        r = np.hanning(2 * ramp + 2)[1:ramp + 1].astype(np.float32)
        w[:ramp] = r
        w[-ramp:] = r[::-1]
    return np.maximum(np.outer(w, w), 1e-3)


def _positions(n: int, tile: int, stride: int) -> list[int]:
    if n <= tile:
        return [0]
    ps = list(range(0, n - tile + 1, stride))
    if ps[-1] != n - tile:
        ps.append(n - tile)
    return ps


def _tta_forward(model, x, gsd, tta: bool):
    """Average over the 8 dihedral transforms, inverting each afterwards."""
    if not tta:
        return model(x, gsd)
    acc, n = None, 0
    outs = []
    for k in range(4):
        for flip in (False, True):
            xi = torch.rot90(x, k, dims=(-2, -1))
            if flip:
                xi = torch.flip(xi, dims=(-1,))
            o = model(xi, gsd)
            h = o["height"]
            s = o.get("sigma")
            if flip:
                h = torch.flip(h, dims=(-1,))
                s = torch.flip(s, dims=(-1,)) if s is not None else None
            h = torch.rot90(h, -k, dims=(-2, -1))
            s = torch.rot90(s, -k, dims=(-2, -1)) if s is not None else None
            outs.append((h, s))
    h = torch.stack([o[0] for o in outs]).mean(0)
    sig = None
    if outs[0][1] is not None:
        sig = torch.stack([o[1] for o in outs]).mean(0)
    spread = torch.stack([o[0] for o in outs]).std(0)
    return {"height": h, "sigma": sig, "tta_spread": spread}


@torch.no_grad()
def predict_scene(model, image: np.ndarray, gsd: float, *, tile: int = 518,
                  overlap: float = 0.25, batch: int = 8, tta: bool = False,
                  level: bool = True, device=None, patch: int = 14,
                  progress: bool = True) -> InferResult:
    """image: (H,W,3) uint8 or float.  gsd: metres/pixel of the input."""
    device = device or next(model.parameters()).device
    model.eval()
    H, W = image.shape[:2]
    tile = max(patch, (tile // patch) * patch)
    ramp = max(1, int(tile * overlap) // 2)
    stride = max(patch, tile - int(tile * overlap))

    if H * W > 40_000_000:
        print(f"  warning: {H}x{W} scene held in memory; ~{H*W*4*3/1e9:.1f} GB of accumulators")

    # ---- 1. coarse whole-scene pass -------------------------------------------------
    href = None
    if level and (H > tile or W > tile):
        scale = max(H, W) / tile
        ch = max(patch, int(round(H / scale / patch)) * patch)
        cw = max(patch, int(round(W / scale / patch)) * patch)
        t = torch.from_numpy(_normalise(image)).unsqueeze(0).to(device)
        small = F.interpolate(t, (ch, cw), mode="bilinear", align_corners=False, antialias=True)
        # the coarse pass sees a coarser ground sample distance, and the model is
        # conditioned on it -- this is where GSD conditioning stops being theoretical
        gsd_coarse = torch.tensor([gsd * (H / ch)], dtype=torch.float32, device=device)
        oc = _tta_forward(model, small, gsd_coarse, False)
        href = F.interpolate(oc["height"].float(), (H, W), mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()

    # ---- 2/3/4. tiles, levelling, blending ------------------------------------------
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    snum = np.zeros((H, W), np.float32)
    sq = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    win = _blend_window(tile, ramp)
    has_sigma = False

    pad_b = max(0, tile - H)
    pad_r = max(0, tile - W)
    img = image
    if pad_b or pad_r:
        img = np.pad(image, ((0, pad_b), (0, pad_r), (0, 0)), mode="reflect")
    Hp, Wp = img.shape[:2]
    ys = _positions(Hp, tile, stride)
    xs = _positions(Wp, tile, stride)
    coords = [(y, x) for y in ys for x in xs]
    gsd_t = torch.tensor([gsd], dtype=torch.float32, device=device)

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        crops = np.stack([_normalise(img[y:y + tile, x:x + tile]) for y, x in chunk])
        xb = torch.from_numpy(crops).to(device)
        o = _tta_forward(model, xb, gsd_t.expand(len(chunk)), tta)
        hb = o["height"].float().cpu().numpy()[:, 0]
        sb = o["sigma"].float().cpu().numpy()[:, 0] if o.get("sigma") is not None else None
        if sb is not None:
            has_sigma = True
        for j, (y, x) in enumerate(chunk):
            h = hb[j]
            ys0, xs0 = y, x
            ye, xe = min(y + tile, H), min(x + tile, W)
            hh, ww = ye - ys0, xe - xs0
            if hh <= 0 or ww <= 0:
                continue
            hcrop = h[:hh, :ww]
            wcrop = win[:hh, :ww]
            if href is not None:
                d = href[ys0:ye, xs0:xe] - hcrop
                hcrop = hcrop + float(np.median(d))
            num[ys0:ye, xs0:xe] += hcrop * wcrop
            den[ys0:ye, xs0:xe] += wcrop
            sq[ys0:ye, xs0:xe] += (hcrop ** 2) * wcrop
            cnt[ys0:ye, xs0:xe] += wcrop
            if sb is not None:
                snum[ys0:ye, xs0:xe] += sb[j][:hh, :ww] * wcrop
        if progress and (i // max(batch, 1)) % 10 == 0:
            print(f"  tiles {min(i+batch, len(coords))}/{len(coords)}", flush=True)

    den = np.maximum(den, 1e-6)
    agl = (num / den).astype(np.float32)
    # where overlapping tiles disagree, confidence is low -- this costs nothing, the
    # data is already there
    var = np.maximum(sq / np.maximum(cnt, 1e-6) - agl ** 2, 0.0)
    disagree = np.sqrt(var).astype(np.float32)
    sigma = (snum / den).astype(np.float32) if has_sigma else None
    if sigma is not None:
        sigma = np.sqrt(sigma ** 2 + disagree ** 2).astype(np.float32)

    return InferResult(
        agl=np.clip(agl, 0, None), sigma=sigma, overlap_disagreement=disagree,
        meta={"tile": tile, "overlap": overlap, "stride": stride, "tta": bool(tta),
              "levelled": bool(href is not None), "n_tiles": len(coords),
              "gsd_m": float(gsd), "scene": [int(H), int(W)]})
