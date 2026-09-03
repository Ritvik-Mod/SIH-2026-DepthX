#!/usr/bin/env python
"""IDEA 1 -- guided / joint-bilateral filtering of the predicted height field.

output = mean(a)*I + mean(b),  a = cov(I,p)/(var(I)+eps),  b = mean(p) - a*mean(I)

Transfers edge structure from a GUIDE image onto the height field.  Twenty lines of box
filters; no OpenCV contrib needed.

The failure mode to watch for is TEXTURE COPYING.  A guided filter carves the guide's
edges into the target whether or not they are height discontinuities, and aerial imagery
is full of strong albedo edges with zero height change: road markings, roof paint,
shadow boundaries, the waterline.  Those become walls.

Hence --guide sem, which uses the semantic head's building probability instead of
luminance.  Semantic boundaries ARE height discontinuities; paint is not.  --guide gray
is the classic version, kept so the two can be compared rather than assumed.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.ndimage import uniform_filter
from _sharpen_common import build, forward_batch, evaluate, row_str, header
from heightmap.data.classes import BUILDING


def guided_filter(p, I, r=4, eps=1e-3):
    """p = signal to filter (height), I = guide, both (H,W) float. r in pixels."""
    k = 2 * r + 1
    mean_I = uniform_filter(I, k); mean_p = uniform_filter(p, k)
    corr_I = uniform_filter(I * I, k); corr_Ip = uniform_filter(I * p, k)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return uniform_filter(a, k) * I + uniform_filter(b, k)


def make_guide(rec, kind):
    if kind == "gray":
        g = (rec["rgb01"] * np.array([0.299, 0.587, 0.114])).sum(-1)
    elif kind == "sem":
        if "sem_logits" not in rec:
            sys.exit("--guide sem needs a checkpoint with the semantic head (A4 onward)")
        z = rec["sem_logits"] - rec["sem_logits"].max(0, keepdims=True)
        e = np.exp(z); g = (e / e.sum(0, keepdims=True))[BUILDING]
    elif kind == "both":
        gray = (rec["rgb01"] * np.array([0.299, 0.587, 0.114])).sum(-1)
        z = rec["sem_logits"] - rec["sem_logits"].max(0, keepdims=True)
        e = np.exp(z); prob = (e / e.sum(0, keepdims=True))[BUILDING]
        g = 0.5 * gray + 0.5 * prob
    else:
        raise ValueError(kind)
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--guide", default="sem", choices=["gray", "sem", "both"])
    ap.add_argument("--radius", nargs="*", type=int, default=[2, 4, 8])
    ap.add_argument("--eps", nargs="*", type=float, default=[1e-4, 1e-3, 1e-2])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    model, cfg, ds, dev = build(a.ckpt, a.root, a.split, a.n, a.device)
    print(f"{len(ds)} tiles from {a.split}   guide={a.guide}\n")
    recs = [forward_batch(model, ds, i, dev, use_gsd=bool(cfg.model.use_gsd))
            for i in range(len(ds))]
    guides = [make_guide(r, a.guide) for r in recs]
    gts = [r["gt"] for r in recs]; masks = [r["mask"] for r in recs]
    clss = [r["cls"] for r in recs]

    print(header())
    base = evaluate([r["pred"] for r in recs], gts, masks, clss)
    print(row_str("baseline (no filter)", base))
    best = None
    for r_ in a.radius:
        for e_ in a.eps:
            out = [guided_filter(rec["pred"].astype(np.float64), g, r_, e_).astype(np.float32)
                   for rec, g in zip(recs, guides)]
            res = evaluate(out, gts, masks, clss)
            print(row_str(f"guided r={r_} eps={e_:g}", res))
            score = res["sharpness"] - 0.0
            if best is None or (res["building"]["mae"] <= base["building"]["mae"] * 1.02
                                and score > best[0]):
                best = (score, r_, e_, res)
    print("-" * 118)
    if best:
        print(f"best sharpness that keeps building MAE within 2% of baseline: "
              f"r={best[1]} eps={best[2]:g}  sharp {best[3]['sharpness']:.3f} "
              f"(baseline {base['sharpness']:.3f})")
    print("\nIf sharpness barely moves, the guide is not carrying the edges you need.\n"
          "If sharpness rises but building MAE or 30m+ bias worsens, that is texture\n"
          "copying: the filter built walls where the guide had contrast and the truth\n"
          "did not.")


if __name__ == "__main__":
    main()
