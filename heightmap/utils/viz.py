from __future__ import annotations
import numpy as np


def colourise(h: np.ndarray, vmin=None, vmax=None, cmap="magma") -> np.ndarray:
    """Height raster -> uint8 RGB, for previews only.  Never for data interchange."""
    import matplotlib
    finite = np.isfinite(h)
    if not finite.any():
        return np.zeros(h.shape + (3,), np.uint8)
    lo = np.percentile(h[finite], 1) if vmin is None else vmin
    hi = np.percentile(h[finite], 99) if vmax is None else vmax
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    x = np.clip((h - lo) / (hi - lo), 0, 1)
    rgb = (matplotlib.colormaps[cmap](x)[..., :3] * 255).astype(np.uint8)
    rgb[~finite] = 0
    return rgb


def panel(rgb, pred, gt=None, err=None) -> np.ndarray:
    """Side-by-side debug panel: input | prediction | truth | error."""
    tiles = [rgb if rgb.dtype == np.uint8 else (rgb * 255).astype(np.uint8)]
    vmax = None
    if gt is not None:
        vmax = float(np.percentile(gt[np.isfinite(gt)], 99)) if np.isfinite(gt).any() else None
    tiles.append(colourise(pred, 0, vmax))
    if gt is not None:
        tiles.append(colourise(gt, 0, vmax))
    if err is not None:
        tiles.append(colourise(np.abs(err), 0, None, "viridis"))
    h = min(t.shape[0] for t in tiles)
    return np.concatenate([t[:h] for t in tiles], axis=1)
