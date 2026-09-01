#!/usr/bin/env python
"""A0 -- the baseline that justifies the whole project.

Runs STOCK Depth Anything V2 on GAMUS tiles and scores it against the LiDAR nDSM after
fitting the best possible affine map from prediction to truth.  That fit is deliberately
generous: it hands the model the perfect calibration for free, so the result is an upper
bound on what any post-hoc rescaling of DAv2 could achieve.

Two fits are reported:
  global   one (a, b) for the whole set  -- what a real deployment could do
  oracle   a fresh (a, b) per tile       -- an upper bound nobody could reach in practice

If the oracle number is poor, no calibration strategy can rescue zero-shot DAv2 and the
head must be replaced, which is the argument the rest of the pipeline rests on.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForDepthEstimation

from heightmap.data.gamus import index_tiles
from heightmap.data.transforms import normalise
from heightmap.metrics import MetricAccumulator, format_table
from heightmap.utils.misc import pick_device


def affine_fit(p, g):
    """least squares a*p + b ~= g"""
    A = np.stack([p, np.ones_like(p)], 1)
    sol, *_ = np.linalg.lstsq(A, g, rcond=None)
    return float(sol[0]), float(sol[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--checkpoint", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--max-tiles", type=int, default=200)
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--out", default="outputs/a0_zero_shot.json")
    a = ap.parse_args()

    dev = pick_device()
    model = AutoModelForDepthEstimation.from_pretrained(a.checkpoint).to(dev).eval()
    tiles = index_tiles(a.root, a.split)[:a.max_tiles]
    if not tiles:
        sys.exit(f"no tiles under {a.root}/*/{a.split}")
    print(f"{len(tiles)} tiles from {a.split}, model={a.checkpoint}, device={dev}")

    preds, gts, sems = [], [], []
    oracle = MetricAccumulator()
    c = a.crop
    for city, tid in tqdm(tiles):
        with h5py.File(f"{a.root}/images/{a.split}/{tid}_RGB.h5") as f:
            rgb = np.asarray(f["image"][()])
        with h5py.File(f"{a.root}/heights/{a.split}/{tid}_AGL.h5") as f:
            agl = np.asarray(f["image"][()], np.float32)
        with h5py.File(f"{a.root}/classes/{a.split}/{tid}_CLS.h5") as f:
            cls = np.asarray(f["image"][()]).astype(np.int64)
        H = rgb.shape[0]; o = (H - c) // 2
        rgb, agl, cls = rgb[o:o+c, o:o+c], agl[o:o+c, o:o+c], cls[o:o+c, o:o+c]

        x = torch.from_numpy(normalise(rgb).transpose(2, 0, 1)).unsqueeze(0).to(dev)
        with torch.no_grad():
            d = model(pixel_values=x).predicted_depth
        if d.ndim == 3:
            d = d.unsqueeze(1)
        d = F.interpolate(d.float(), (c, c), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        m = np.isfinite(agl) & np.isfinite(d)
        p, g = d[m].astype(np.float64), agl[m].astype(np.float64)
        if p.size < 100 or p.std() < 1e-8:
            continue
        aa, bb = affine_fit(p, g)                       # per-tile oracle
        oracle.update(np.clip(aa * d + bb, 0, None), agl, m, cls)
        idx = np.random.default_rng(0).integers(0, p.size, min(p.size, 20000))
        preds.append(p[idx]); gts.append(g[idx])
        sems.append(cls[m][idx])

    P, G, S = np.concatenate(preds), np.concatenate(gts), np.concatenate(sems)
    ga, gb = affine_fit(P, G)
    glob = MetricAccumulator()
    fit = np.clip(ga * P + gb, 0, None).astype(np.float32)
    glob.update(fit[None, None], G.astype(np.float32)[None, None],
                np.ones((1, 1, len(G)), bool), S.astype(np.int64)[None])

    res = {"model": a.checkpoint, "split": a.split, "n_tiles": len(tiles),
           "global_affine": {"a": ga, "b": gb},
           "global_fit": glob.result(), "per_tile_oracle": oracle.result()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, default=float)

    print(f"\nGLOBAL affine fit  h = {ga:.4f} * dav2 + {gb:.3f}")
    print(format_table(res["global_fit"]))
    print(f"\nPER-TILE ORACLE (upper bound, unreachable in deployment)")
    print(format_table(res["per_tile_oracle"]))
    print(f"\nwritten to {a.out}")
    print("\nInterpretation: this is the ceiling for any post-hoc rescaling of stock DAv2.\n"
          "Compare against the fine-tuned model (A1+) to quantify what replacing the head buys.")


if __name__ == "__main__":
    main()
