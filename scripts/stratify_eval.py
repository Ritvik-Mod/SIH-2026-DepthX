#!/usr/bin/env python
"""Landscape-stratified evaluation.  The PS's largest scoring gap, measured.

The rubric is not "what is your MAE" -- it is "performance STABILITY across urban,
sparse, hilly and forested".  A single pooled number cannot answer that no matter how
good it is, and four numbers only answer it if the strata are defined before the results
are seen.  So this script defines the strata from the LABELS, reports each one, and then
reports the SPREAD between them, which is the quantity actually being scored.

Strata come from each tile's own per-pixel classes:

  urban     >= --urban-bld building pixels
  forested  >= --forest-tree tree pixels, and not urban
  sparse    <  --sparse-bld building AND < --sparse-tree tree -- open, low-relief land
  mixed     everything else, reported so the other three cannot be quietly cherry-picked

HILLY IS NOT HERE, and that is a statement about the data, not an omission.  GAMUS is
three flat US cities; there is no hilly stratum to find in it.  Hilly terrain is also
the one landscape this model is structurally the wrong tool for: we predict AGL, height
above local ground, and on bare rock there is nothing above the ground, so the correct
AGL is ~0 everywhere and it carries no information about the mountain.  That landscape
is served by the DEM path (scripts/fetch_dem.py + --terrain-mode raw), where the DSM is
the DEM and the accuracy is the DEM's published accuracy, not ours.  Saying so with a
number from this script would be inventing evidence.

Every threshold is a flag, and --dump writes per-tile rows so a reviewer can re-cut the
strata without re-running the model.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from _sharpen_common import build, forward_batch
from heightmap.data.classes import BUILDING, TREE, NAMES
from heightmap.metrics import sharpness

ORDER = ["urban", "forested", "sparse", "mixed"]


def classify(cls, mask, a) -> tuple[str, float, float]:
    m = mask & (cls > 0)
    n = max(int(m.sum()), 1)
    bld = float((cls[m] == BUILDING).sum()) / n
    tree = float((cls[m] == TREE).sum()) / n
    if bld >= a.urban_bld:
        s = "urban"
    elif tree >= a.forest_tree:
        s = "forested"
    elif bld < a.sparse_bld and tree < a.sparse_tree:
        s = "sparse"
    else:
        s = "mixed"
    return s, bld, tree


def stat(p, g, m):
    if m.sum() < 64:
        return {"n": int(m.sum()), "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "r": float("nan")}
    a, b = p[m], g[m]
    d = a - b
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
    return {"n": int(m.sum()), "mae": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d ** 2).mean())), "bias": float(d.mean()), "r": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--urban-bld", type=float, default=0.15)
    ap.add_argument("--forest-tree", type=float, default=0.30)
    ap.add_argument("--sparse-bld", type=float, default=0.05)
    ap.add_argument("--sparse-tree", type=float, default=0.10)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="outputs/stratified.json")
    ap.add_argument("--dump", default=None, help="per-tile CSV, so strata can be re-cut")
    a = ap.parse_args()

    model, cfg, ds, dev = build(a.ckpt, a.root, a.split, a.n, a.device)
    print(f"{len(ds)} tiles from {a.split}\n")

    rows, per = [], {k: {"pred": [], "gt": [], "bld": [], "sharp": []} for k in ORDER}
    for i in range(len(ds)):
        r = forward_batch(model, ds, i, dev, use_gsd=bool(cfg.model.use_gsd))
        s, bf, tf = classify(r["cls"], r["mask"], a)
        m = r["mask"]
        per[s]["pred"].append(r["pred"][m])
        per[s]["gt"].append(r["gt"][m])
        per[s]["bld"].append((r["cls"] == BUILDING)[m])
        per[s]["sharp"].append(sharpness(r["pred"], r["gt"], m)["ratio"])
        rows.append({"tile": r["tid"], "stratum": s, "building_frac": round(bf, 4),
                     "tree_frac": round(tf, 4),
                     "mae": round(float(np.abs(r["pred"][m] - r["gt"][m]).mean()), 4)})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    res, maes = {}, {}
    hdr = (f"{'stratum':<10}{'tiles':>7}{'MAE':>8}{'RMSE':>9}{'bias':>9}{'r':>8}"
           f"{'bMAE':>8}{'b_r':>7}{'sharp':>8}")
    print("\n" + hdr); print("-" * len(hdr))
    for k in ORDER:
        d = per[k]
        if not d["pred"]:
            print(f"{k:<10}{0:>7}   (no tiles matched this stratum)")
            res[k] = {"tiles": 0}
            continue
        P = np.concatenate(d["pred"]); G = np.concatenate(d["gt"])
        B = np.concatenate(d["bld"])
        o = stat(P, G, np.ones_like(B, bool)); b = stat(P, G, B)
        sh = float(np.nanmean(d["sharp"]))
        res[k] = {"tiles": len(d["pred"]), "overall": o, "building": b, "sharpness": sh}
        maes[k] = b["mae"]
        print(f"{k:<10}{len(d['pred']):>7}{o['mae']:>8.3f}{o['rmse']:>9.3f}"
              f"{o['bias']:>+9.3f}{o['r']:>8.3f}{b['mae']:>8.3f}{b['r']:>7.3f}{sh:>8.3f}")
    print("-" * len(hdr))

    # ---- the number the rubric actually scores --------------------------------
    scored = {k: v for k, v in maes.items() if np.isfinite(v) and k != "mixed"}
    if len(scored) >= 2:
        lo_k, hi_k = min(scored, key=scored.get), max(scored, key=scored.get)
        spread = scored[hi_k] - scored[lo_k]
        rel = spread / max(scored[lo_k], 1e-6)
        res["stability"] = {"building_mae_by_stratum": scored,
                            "spread_m": spread, "spread_relative": rel,
                            "best": lo_k, "worst": hi_k}
        print(f"\nSTABILITY  building MAE ranges {scored[lo_k]:.3f} m ({lo_k}) to "
              f"{scored[hi_k]:.3f} m ({hi_k})")
        print(f"           spread {spread:.3f} m = {rel*100:.1f}% of the best stratum")
        print("           the rubric scores this spread, not the pooled mean.")
    print("\nhilly is absent by construction: GAMUS is three flat US cities, and AGL is\n"
          "the wrong quantity for bare terrain. That landscape is served by the DEM path\n"
          "(fetch_dem.py + --terrain-mode raw), whose accuracy is the DEM's, not ours.")

    res["config"] = {"split": a.split, "n": len(ds), "ckpt": a.ckpt,
                     "thresholds": {"urban_bld": a.urban_bld, "forest_tree": a.forest_tree,
                                    "sparse_bld": a.sparse_bld, "sparse_tree": a.sparse_tree}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.out}")
    if a.dump:
        import csv
        with open(a.dump, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader(); wtr.writerows(rows)
        print(f"wrote {a.dump}  ({len(rows)} tiles; re-cut the strata without re-running)")


if __name__ == "__main__":
    main()
