#!/usr/bin/env python
"""IDEA 2 -- semantic mask flattening.

Take the predicted building mask, split it into connected instances, and replace each
instance's interior with one height (median) or a fitted plane.  GAMUS buildings are
predominantly flat-roofed, so this converts a melted blob into a slab with vertical
walls -- exactly the shape the sharpness metric is asking for.

Three things to be honest about, all measurable with the flags here:

  * it runs on PREDICTED semantics, so mask errors become geometry errors.  --oracle-sem
    swaps in the ground-truth mask, which is not deployable but bounds how much of any
    loss is the mask's fault rather than the idea's.
  * it is a PRIOR, not inference.  You are asserting roofs are flat.  --mode plane
    relaxes that to "roofs are planar", which is a weaker and often better assumption.
  * the median is taken over blurred pixels.  --erode shrinks each instance before
    measuring so edge pixels, which are contaminated by the very blur being removed, do
    not set the height of the slab.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage as ndi
from _sharpen_common import build, forward_batch, evaluate, row_str, header
from heightmap.data.classes import BUILDING


def flatten(pred, mask_bld, mode="median", min_area=64, erode=2):
    """Replace each connected building instance with a constant or a fitted plane."""
    out = pred.copy()
    lab, n = ndi.label(mask_bld)
    if n == 0:
        return out, 0
    inner = mask_bld
    if erode > 0:
        inner = ndi.binary_erosion(mask_bld, np.ones((3, 3), bool), iterations=erode)
    done = 0
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < min_area:
            continue
        core = comp & inner
        src = core if core.sum() >= max(16, 0.1 * comp.sum()) else comp
        ys, xs = np.nonzero(src)
        v = pred[ys, xs]
        if mode == "median":
            out[comp] = np.median(v)
        else:                                   # least-squares plane z = ax + by + c
            A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(np.float64)
            coef, *_ = np.linalg.lstsq(A, v.astype(np.float64), rcond=None)
            cy, cx = np.nonzero(comp)
            out[comp] = (coef[0] * cx + coef[1] * cy + coef[2]).astype(pred.dtype)
        done += 1
    return out, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--mode", nargs="*", default=["median", "plane"])
    ap.add_argument("--erode", nargs="*", type=int, default=[0, 2, 4])
    ap.add_argument("--min-area", type=int, default=64)
    ap.add_argument("--oracle-sem", action="store_true",
                    help="use the GROUND-TRUTH building mask. Not deployable; it "
                         "separates 'the idea is wrong' from 'the mask is wrong'.")
    ap.add_argument("--flatten-ground", action="store_true",
                    help="also snap non-building, non-tree pixels to their median")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    model, cfg, ds, dev = build(a.ckpt, a.root, a.split, a.n, a.device)
    recs = [forward_batch(model, ds, i, dev, use_gsd=bool(cfg.model.use_gsd))
            for i in range(len(ds))]
    if not a.oracle_sem and "sem" not in recs[0]:
        sys.exit("no semantic head in this checkpoint; use --oracle-sem or an A4+ ckpt")
    gts = [r["gt"] for r in recs]; masks = [r["mask"] for r in recs]
    clss = [r["cls"] for r in recs]
    bmasks = [(r["cls"] == BUILDING) if a.oracle_sem else (r["sem"] == BUILDING)
              for r in recs]
    iou = np.mean([float(((b & (c == BUILDING)).sum()) /
                         max((b | (c == BUILDING)).sum(), 1)) for b, c in zip(bmasks, clss)])
    print(f"{len(ds)} tiles from {a.split}   mask="
          f"{'GROUND TRUTH (oracle)' if a.oracle_sem else 'predicted'}"
          f"   building IoU vs truth {iou:.3f}\n")

    print(header())
    base = evaluate([r["pred"] for r in recs], gts, masks, clss)
    print(row_str("baseline (no flattening)", base))
    for mode in a.mode:
        for er in a.erode:
            outs, tot = [], 0
            for rec, bm in zip(recs, bmasks):
                o, k = flatten(rec["pred"], bm, mode, a.min_area, er)
                if a.flatten_ground:
                    g = (~bm) & (rec["cls"] != 6) & rec["mask"]
                    if g.sum() > 64:
                        o[g] = np.median(rec["pred"][g])
                outs.append(o); tot += k
            res = evaluate(outs, gts, masks, clss)
            print(row_str(f"{mode} erode={er} ({tot} inst)", res))
    print("-" * 118)
    print("\nRead building MAE and 30m+ bias, not sharpness alone.  Flattening ALWAYS\n"
          "raises sharpness -- a slab has infinitely sharp walls.  The question is what\n"
          "it costs.  If --oracle-sem is much better than predicted, the limit is the\n"
          "semantic head, not the idea.")


if __name__ == "__main__":
    main()
