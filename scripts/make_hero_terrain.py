"""Render the homepage backdrop from the model's own output.

    python scripts/make_hero_terrain.py

Reads the Swiss hillside DSM from the packaged samples and writes
terrain3d/public/hero-terrain.jpg: a topographic-map rendering -- stepped layer
tints, hillshade and contour lines. It is real terrain, reconstructed by
DepthWizard from one image and set on the Copernicus DTM, rather than a generic
pattern.

Stepped tints (one flat colour per contour band) rather than a smooth ramp: a
smooth ramp over rolling ground reads as an abstract gradient, whereas bands
read at once as a printed map. The DSM carries every roof and tree crown, which
would turn the contours into noise, so the field is smoothed to landform scale
first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "terrain3d" / "public" / "samples" / "swiss-hillside" / "heightmap.tif"
OUT = ROOT / "terrain3d" / "public" / "hero-terrain.jpg"

W, H = 1800, 1100          # output size
SS = 2                     # supersampling, for anti-aliased contour lines

# Hypsometric tints, low to high: valley teal, sage, pale straw, ochre, clay,
# muted rose, then near-white for the tops. Cartographic ordering, kept soft so
# dark text can sit on top of it.
RAMP = [
    (0.00, (62, 140, 135)),
    (0.18, (118, 176, 140)),
    (0.38, (196, 214, 158)),
    (0.56, (236, 214, 150)),
    (0.72, (226, 170, 118)),
    (0.86, (196, 124, 116)),
    (1.00, (244, 236, 228)),
]


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur via cumulative sums; three passes approximate a Gaussian."""
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        a = (hi - lo) / (2 * r + 1)
    return a


def tint(t: np.ndarray) -> np.ndarray:
    out = np.zeros(t.shape + (3,), np.float32)
    for (t0, c0), (t1, c1) in zip(RAMP, RAMP[1:]):
        m = (t >= t0) & (t <= t1)
        f = ((t[m] - t0) / (t1 - t0))[:, None]
        out[m] = np.array(c0) * (1 - f) + np.array(c1) * f
    return out / 255.0


def main():
    h = np.asarray(Image.open(SRC), dtype=np.float32)

    # landform scale, not roof scale
    for _ in range(3):
        h = box_blur(h, 12)

    # centre crop to the output aspect, then resample to the supersampled size
    ah, aw = h.shape
    want = W / H
    if aw / ah > want:
        cw = int(ah * want)
        h = h[:, (aw - cw) // 2:(aw - cw) // 2 + cw]
    else:
        ch = int(aw / want)
        h = h[(ah - ch) // 2:(ah - ch) // 2 + ch, :]
    h = np.asarray(Image.fromarray(h).resize((W * SS, H * SS), Image.BICUBIC), dtype=np.float32)

    # contour interval; each band gets one flat tint, taken at its mid-height
    step = 4.0
    band = np.floor(h / step)
    lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
    t = np.clip(((band + 0.5) * step - lo) / (hi - lo), 0, 1)

    # hillshade, light from the north-west
    gy, gx = np.gradient(h)
    k = 3.0 / SS                                 # relief emphasis
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(h)
    n = np.sqrt(nx * nx + ny * ny + nz * nz)
    az, el = np.radians(315), np.radians(42)
    lx, ly, lz = np.cos(el) * np.sin(az), -np.cos(el) * np.cos(az), np.sin(el)
    shade = np.clip((nx * lx + ny * ly + nz * lz) / n, 0, 1)

    rgb = tint(t) * (0.7 + 0.42 * shade[..., None])

    # contour lines in the classic map brown; every fifth (20 m) is heavier
    ink = np.array([110, 84, 62]) / 255.0
    edge = np.zeros(h.shape, bool)
    edge[:, 1:] |= band[:, 1:] != band[:, :-1]
    edge[1:, :] |= band[1:, :] != band[:-1, :]
    index = edge & (np.mod(band, 5) == 0)
    heavy = index.copy()
    heavy[1:, :] |= index[:-1, :]
    heavy[:, 1:] |= index[:, :-1]
    rgb = np.where(edge[..., None], rgb * 0.55 + ink * 0.45, rgb)
    rgb = np.where(heavy[..., None], rgb * 0.35 + ink * 0.65, rgb)

    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((W, H), Image.LANCZOS)
    img.save(OUT, "JPEG", quality=84, optimize=True, progressive=True)
    print(f"wrote {OUT.relative_to(ROOT)}  {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
