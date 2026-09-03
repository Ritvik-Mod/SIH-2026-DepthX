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

from heightmap.data.gamus import index_tiles, IMAGE_KINDS
from heightmap.losses.shadow import predicted_shadow, observed_shadow

CFG = OmegaConf.create(dict(tau=1.0, tau_dark=0.35, sigma_s=0.08, local_contrast=True,
                            bg_ksize=65, tau_rel=0.05, use_blue=True, tau_blue=0.02,
                            sigma_blue=0.03, max_steps=16, max_dist_px=150.0))


@torch.no_grad()
def score_batch(h, obs, gsd, elevs, azims, dev):
    """IoU for every (elev, azim) candidate at once.

    The search is ~50 candidates per tile.  Evaluating them one at a time meant ~800
    grid_sample launches per tile and 2.3 s on CPU -- over three hours for the split.
    predicted_shadow is already batched over samples, so broadcast the single height
    field across the candidates and do it in one call.
    """
    n = len(elevs)
    hb = h.expand(n, -1, -1, -1)                       # view, not a copy
    soft, rel = predicted_shadow(
        hb, torch.full((n,), gsd, device=dev),
        torch.tensor(elevs, dtype=torch.float32, device=dev),
        torch.tensor(azims, dtype=torch.float32, device=dev),
        tau=CFG.tau, max_steps=CFG.max_steps, max_dist_px=CFG.max_dist_px)
    p = (soft > 0.5) & rel
    o = (obs > 0.5) & rel
    inter = (p & o).flatten(1).sum(1).float()
    union = (p | o).flatten(1).sum(1).float().clamp(min=1.0)
    return (inter / union).cpu().numpy()


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
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available())
                       else ("cuda" if a.device == "cuda" else "cpu"))
    print(f"device: {dev}")
    out, skipped = {}, 0
    for split in a.splits:
        tiles = index_tiles(a.root, split)
        if a.max_tiles:
            tiles = tiles[:a.max_tiles]
        for city, tid in tqdm(tiles, desc=split):
            with h5py.File(f"{a.root}/heights/{split}/{tid}_AGL.h5") as f:
                agl = np.asarray(f["image"][()], np.float32)
            img_path = None
            for kind in IMAGE_KINDS:               # DC/PHL use _RGB, NYC uses _IMG
                cand = f"{a.root}/images/{split}/{tid}_{kind}.h5"
                if os.path.exists(cand):
                    img_path = cand; break
            if img_path is None:
                skipped += 1; continue
            with h5py.File(img_path) as f:
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

            # coarse sweep, all candidates in one batched call
            cand = [(float(el), float(az)) for az in range(0, 360, 30)
                    for el in (25.0, 40.0, 55.0)]
            v = score_batch(h, obs, a.gsd, [c[0] for c in cand], [c[1] for c in cand], dev)
            i = int(v.argmax()); best = (float(v[i]), cand[i][0], cand[i][1])
            el0, az0 = best[1], best[2]

            cand = [(float(el), float(az % 360))
                    for az in np.arange(az0 - 20, az0 + 21, 10)
                    for el in np.arange(max(10, el0 - 12), min(70, el0 + 13), 6)]
            v = score_batch(h, obs, a.gsd, [c[0] for c in cand], [c[1] for c in cand], dev)
            i = int(v.argmax())
            if float(v[i]) > best[0]:
                best = (float(v[i]), cand[i][0], cand[i][1])

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
