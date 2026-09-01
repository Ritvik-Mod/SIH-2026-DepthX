#!/usr/bin/env python
"""Recover per-tile sun elevation and azimuth for GAMUS, which ships no metadata.

GAMUS has ground-truth AGL, so the sun angle can be recovered by rendering the shadow
that height field WOULD cast at each candidate angle and picking the angle whose
rendered shadow best matches the shadows actually visible in the image.  Coarse-to-fine
grid search.

This is what makes the shadow-consistency ablation (A6) runnable on GAMUS at all.
Angles are cached to JSON and consumed via data.sun_cache.

Note: these are ESTIMATES, not metadata.  They are good enough to supervise a soft
regulariser; do not present them as recorded acquisition parameters.
"""
import sys, os, json, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

from heightmap.data.gamus import index_tiles
from heightmap.losses.shadow import predicted_shadow, observed_shadow

CFG = OmegaConf.create(dict(tau=1.0, tau_dark=0.35, sigma_s=0.08, local_contrast=True,
                            bg_ksize=65, tau_rel=0.05, use_blue=True, tau_blue=0.02,
                            sigma_blue=0.03, max_steps=16, max_dist_px=150.0))


def score(h, obs, gsd, elev, azim, dev):
    soft, rel = predicted_shadow(h, torch.tensor([gsd], device=dev),
                                 torch.tensor([elev], device=dev),
                                 torch.tensor([azim], device=dev),
                                 tau=CFG.tau, max_steps=CFG.max_steps,
                                 max_dist_px=CFG.max_dist_px)
    p = (soft > 0.5) & rel
    o = (obs > 0.5) & rel
    inter = float((p & o).sum()); union = float((p | o).sum())
    return inter / max(union, 1.0)          # IoU; higher is better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--gsd", type=float, default=0.33)  # GAMUS paper sec.1
    ap.add_argument("--size", type=int, default=512, help="centre crop used for the search")
    ap.add_argument("--max-tiles", type=int, default=0)
    ap.add_argument("--min-iou", type=float, default=0.12,
                    help="below this the estimate is untrustworthy and the tile is skipped")
    ap.add_argument("--out", default="data/GAMUS/sun_angles.json")
    a = ap.parse_args()

    dev = torch.device("cpu")   # grid search is memory-bound, not compute-bound
    out, skipped = {}, 0
    for split in a.splits:
        tiles = index_tiles(a.root, split)
        if a.max_tiles:
            tiles = tiles[:a.max_tiles]
        for city, tid in tqdm(tiles, desc=split):
            with h5py.File(f"{a.root}/heights/{split}/{tid}_AGL.h5") as f:
                agl = np.asarray(f["image"][()], np.float32)
            with h5py.File(f"{a.root}/images/{split}/{tid}_RGB.h5") as f:
                rgb = np.asarray(f["image"][()])
            H = agl.shape[0]; o = max(0, (H - a.size) // 2); s = min(a.size, H)
            agl, rgb = agl[o:o+s, o:o+s], rgb[o:o+s, o:o+s]
            if not np.isfinite(agl).all() or agl.max() < 3.0:
                skipped += 1; continue

            h = torch.from_numpy(np.nan_to_num(agl)).float()[None, None].to(dev)
            im = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(dev) / 255.0
            obs = observed_shadow(im, CFG.tau_dark, CFG.sigma_s, CFG.local_contrast,
                                  CFG.bg_ksize, CFG.tau_rel, CFG.use_blue,
                                  CFG.tau_blue, CFG.sigma_blue)

            best = (-1.0, None, None)
            for az in range(0, 360, 30):                       # coarse
                for el in (25, 40, 55):
                    v = score(h, obs, a.gsd, el, float(az), dev)
                    if v > best[0]:
                        best = (v, el, float(az))
            _, el0, az0 = best
            for az in np.arange(az0 - 20, az0 + 21, 10):        # fine
                for el in np.arange(max(10, el0 - 12), min(70, el0 + 13), 6):
                    v = score(h, obs, a.gsd, float(el), float(az % 360), dev)
                    if v > best[0]:
                        best = (v, float(el), float(az % 360))

            iou, el, az = best
            if iou < a.min_iou:
                skipped += 1; continue
            out[tid] = [round(float(el), 2), round(float(az), 2)]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nestimated {len(out)} tiles, skipped {skipped} (flat or low agreement)")
    if out:
        e = np.array([v[0] for v in out.values()]); z = np.array([v[1] for v in out.values()])
        print(f"  elevation  p10 {np.percentile(e,10):.1f}  p50 {np.percentile(e,50):.1f}"
              f"  p90 {np.percentile(e,90):.1f}")
        print(f"  azimuth    p10 {np.percentile(z,10):.1f}  p50 {np.percentile(z,50):.1f}"
              f"  p90 {np.percentile(z,90):.1f}")
    print(f"written to {a.out}   -> set data.sun_cache={a.out}")


if __name__ == "__main__":
    main()
