#!/usr/bin/env python
"""Predict vs ground truth, side by side, for a handful of GAMUS tiles.

Four panels per tile: RGB, ground-truth AGL, predicted AGL, and the signed error.
Prediction and truth share ONE colour scale -- rendering them on independent scales
makes any model look right, because the eye reads relative structure and not metres.
The error panel is diverging and symmetric, so over- and under-prediction are visually
distinguishable rather than both reading as "wrong".

Defaults to the val split.  Predicting on train tiles is showing the model its own
homework: it will look far better than it is, so --split train prints a warning.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heightmap.data.gamus import index_tiles, IMAGE_KINDS
from heightmap.data.classes import BUILDING
from heightmap.predict import load_model
from heightmap.infer import predict_scene
from heightmap.utils.misc import pick_device


def load_tile(root, split, tid):
    with h5py.File(f"{root}/heights/{split}/{tid}_AGL.h5") as f:
        agl = np.asarray(f["image"][()], np.float32)
    with h5py.File(f"{root}/classes/{split}/{tid}_CLS.h5") as f:
        cls = np.asarray(f["image"][()])
    for kind in IMAGE_KINDS:                       # DC/PHL _RGB, NYC _IMG
        p = f"{root}/images/{split}/{tid}_{kind}.h5"
        if os.path.exists(p):
            with h5py.File(p) as f:
                return np.asarray(f["image"][()]), agl, cls
    return None, agl, cls


def stats(pred, gt, mask):
    """MAE / bias / correlation over mask.  Returns nan where the mask is too small."""
    p, g = pred[mask], gt[mask]
    if p.size < 16:
        return dict(n=int(p.size), mae=np.nan, bias=np.nan, r=np.nan)
    d = p - g
    r = np.nan
    if p.std() > 1e-6 and g.std() > 1e-6:
        r = float(np.corrcoef(p, g)[0, 1])
    return dict(n=int(p.size), mae=float(np.abs(d).mean()), bias=float(d.mean()), r=r)


def render(tid, rgb, gt, pred, cls, gsd, out_path, split):
    finite = np.isfinite(gt)
    valid = finite & (gt > -100.0)
    bld = valid & (cls == BUILDING)
    ov, bs = stats(pred, gt, valid), stats(pred, gt, bld)

    # ONE scale for truth and prediction.  p99 of TRUTH, so the prediction cannot
    # quietly set its own range and look better than it is.
    vmax = max(float(np.percentile(gt[valid], 99)) if valid.any() else 1.0, 1.0)
    err = np.where(valid, pred - gt, np.nan)
    elim = max(float(np.nanpercentile(np.abs(err), 95)) if valid.any() else 1.0, 0.5)

    fig, ax = plt.subplots(1, 4, figsize=(21, 5.6))
    fig.patch.set_facecolor("white")
    ax[0].imshow(rgb); ax[0].set_title("RGB input", fontsize=12, fontweight="bold")
    ax[1].imshow(np.where(valid, gt, 0), cmap="viridis", vmin=0, vmax=vmax)
    ax[1].set_title("Ground truth AGL (LiDAR)", fontsize=12, fontweight="bold")
    im = ax[2].imshow(np.where(valid, pred, 0), cmap="viridis", vmin=0, vmax=vmax)
    ax[2].set_title("Predicted AGL — same colour scale", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03).set_label("metres AGL", fontsize=9)
    ie = ax[3].imshow(err, cmap="RdBu_r", vmin=-elim, vmax=elim)
    ax[3].set_title("Error (pred − truth)\nred = too tall, blue = too short",
                    fontsize=12, fontweight="bold")
    fig.colorbar(ie, ax=ax[3], fraction=0.046, pad=0.03).set_label("metres", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    sub = (f"{tid}  ·  split={split}  ·  {gt.shape[1]}×{gt.shape[0]} px at {gsd} m/px\n"
           f"ALL pixels   MAE {ov['mae']:.2f} m   bias {ov['bias']:+.2f} m   r {ov['r']:.3f}"
           f"        BUILDINGS   MAE {bs['mae']:.2f} m   bias {bs['bias']:+.2f} m   r {bs['r']:.3f}"
           f"   ({100*bs['n']/max(ov['n'],1):.1f}% of pixels)\n"
           f"truth p50 {np.percentile(gt[valid],50):.2f} / p99 {np.percentile(gt[valid],99):.1f} m"
           f"      pred p50 {np.percentile(pred[valid],50):.2f} / p99 {np.percentile(pred[valid],99):.1f} m")
    fig.suptitle(sub, fontsize=10.5, y=0.995, linespacing=1.45)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.subplots_adjust(top=0.78, wspace=0.06)
    fig.savefig(out_path, dpi=105, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(tile=tid, overall=ov, building=bs,
                truth_p50=float(np.percentile(gt[valid], 50)),
                pred_p50=float(np.percentile(pred[valid], 50)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--per-city", type=int, default=1)
    ap.add_argument("--gsd", type=float, default=0.33)
    ap.add_argument("--out", default="outputs/compare")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    if a.split == "train":
        print("WARNING: train tiles were used to fit these weights. The comparison will\n"
              "         flatter the model. Use --split val or test for an honest look.\n")
    os.makedirs(a.out, exist_ok=True)
    dev = pick_device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    print(f"device {dev}   backbone {cfg.model.checkpoint.split('/')[-1]}   head {cfg.model.head}")

    tiles = index_tiles(a.root, a.split)
    if a.tiles:
        chosen = [(t.split("_")[0], t) for t in a.tiles]
    else:
        by = {}
        for city, tid in tiles:
            by.setdefault(city, []).append(tid)
        chosen = [(c, t) for c in sorted(by) for t in by[c][:a.per_city]]

    rows = []
    for city, tid in chosen:
        rgb, gt, cls = load_tile(a.root, a.split, tid)
        if rgb is None:
            print(f"skip {tid}: no image"); continue
        res = predict_scene(model, rgb, a.gsd, tile=int(cfg.data.crop),
                            overlap=float(cfg.infer.overlap), batch=4,
                            tta=a.tta, device=dev, progress=False)
        out = os.path.join(a.out, f"{tid}_compare.png")
        rows.append(render(tid, rgb, gt, res.agl, cls, a.gsd, out, a.split))
        r = rows[-1]
        print(f"{tid:14s} all MAE {r['overall']['mae']:.2f} bias {r['overall']['bias']:+.2f} "
              f"r {r['overall']['r']:.3f} | bldg MAE {r['building']['mae']:.2f} "
              f"bias {r['building']['bias']:+.2f} r {r['building']['r']:.3f} "
              f"| p50 truth {r['truth_p50']:.2f} pred {r['pred_p50']:.2f}  -> {out}")

    if rows:
        m = np.mean([r["building"]["mae"] for r in rows])
        print(f"\nmean building MAE over {len(rows)} tiles: {m:.3f} m")
        with open(os.path.join(a.out, "compare.json"), "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
