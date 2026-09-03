#!/usr/bin/env python
"""Measure the dataset instead of assuming it.

Determines h_max (clip threshold), long-tail severity, per-city differences and class
frequencies.  These decide bin range, loss weights and whether the SILog +1 shift is
safe -- all of which are otherwise guesses.  Twenty minutes of compute, and it becomes
a slide.
"""
import sys, os, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py
from tqdm import tqdm
from heightmap.data.gamus import index_tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--splits", nargs="*", default=["train"])
    ap.add_argument("--max-tiles", type=int, default=0, help="0 = all")
    ap.add_argument("--sample-px", type=int, default=40000, help="pixels sampled per tile")
    ap.add_argument("--out", default="outputs/dataset_stats.json")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    per_city = collections.defaultdict(list)
    cls_counts = collections.Counter()
    n_neg = n_nan = n_px = 0
    hi = []

    for split in a.splits:
        tiles = index_tiles(a.root, split)
        if a.max_tiles:
            tiles = tiles[:a.max_tiles]
        if not tiles:
            print(f"no tiles in {a.root}/*/{split}"); continue
        for city, tid in tqdm(tiles, desc=split):
            with h5py.File(f"{a.root}/heights/{split}/{tid}_AGL.h5") as f:
                h = np.asarray(f["image"][()], np.float32)
            with h5py.File(f"{a.root}/classes/{split}/{tid}_CLS.h5") as f:
                c = np.asarray(f["image"][()]).astype(np.int64).ravel()
            n_px += h.size
            n_nan += int((~np.isfinite(h)).sum())
            fin = h[np.isfinite(h)]
            n_neg += int((fin < 0).sum())
            hi.append(float(fin.max()) if fin.size else 0.0)
            s = fin.ravel()
            if s.size > a.sample_px:
                s = s[rng.integers(0, s.size, a.sample_px)]
            per_city[city].append(s)
            cls_counts.update(np.bincount(c, minlength=16).tolist().__iter__() and
                              dict(zip(*np.unique(c, return_counts=True))))

    if not per_city:
        sys.exit("nothing measured")

    allv = np.concatenate([np.concatenate(v) for v in per_city.values()])
    def prof(x):
        return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)),
                "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
                "p99.9": float(np.percentile(x, 99.9)), "max": float(x.max()),
                "frac_le_0.5m": float((x <= 0.5).mean()),
                "frac_ge_30m": float((x >= 30).mean())}

    out = {"splits": a.splits, "root": a.root,
           "overall": prof(allv),
           "per_city": {c: prof(np.concatenate(v)) for c, v in per_city.items()},
           "integrity": {"n_pixels_scanned": int(n_px), "n_nonfinite": n_nan,
                         "n_negative": n_neg, "max_tile_height": float(max(hi))},
           "class_frequencies": {int(k): int(v) for k, v in sorted(cls_counts.items())}}

    tail = allv.max()
    rec_hmax = float(np.ceil(max(np.percentile(allv, 99.99), tail * 0.6) / 10) * 10)
    out["recommendations"] = {
        "data.h_max": rec_hmax,
        "silog_shift_safe": bool(n_neg == 0 and n_nan == 0),
        "note": ("SILog uses log(h+1); safe only if heights are non-negative and finite. "
                 "If n_negative>0, mask those pixels or raise the shift.")}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)

    o = out["overall"]
    print(f"\n{'':14s}{'p50':>8}{'p90':>8}{'p99':>8}{'p99.9':>9}{'max':>9}{'<=0.5m':>9}{'>=30m':>8}")
    print(f"{'ALL':14s}{o['p50']:>8.2f}{o['p90']:>8.2f}{o['p99']:>8.2f}{o['p99.9']:>9.2f}"
          f"{o['max']:>9.2f}{o['frac_le_0.5m']:>9.3f}{o['frac_ge_30m']:>8.3f}")
    for c, p in out["per_city"].items():
        print(f"{c:14s}{p['p50']:>8.2f}{p['p90']:>8.2f}{p['p99']:>8.2f}{p['p99.9']:>9.2f}"
              f"{p['max']:>9.2f}{p['frac_le_0.5m']:>9.3f}{p['frac_ge_30m']:>8.3f}")
    print(f"\nnon-finite {n_nan}  negative {n_neg}")
    print(f"classes seen: {sorted(out['class_frequencies'])}")
    print(f"\nRECOMMEND  data.h_max={rec_hmax}   silog +1 shift safe: "
          f"{out['recommendations']['silog_shift_safe']}")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
