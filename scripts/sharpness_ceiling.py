#!/usr/bin/env python
"""What sharpness score is actually ACHIEVABLE?

Every sharpening attempt has landed near 0.13 and none moved it.  Before trying more,
calibrate the metric: degrade the GROUND TRUTH in known ways and score it against
itself.  That gives a noise floor and a ceiling, and turns 0.13 from a bare number into
something with a scale attached.

If a one-pixel shift of the truth already scores 0.4, then 0.13 is genuinely poor and
worth more work.  If it scores 0.18, the metric saturates almost immediately under
sub-pixel misregistration and 0.13 is near the practical ceiling -- in which case the
honest move is to stop optimising it and say so.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py
from scipy.ndimage import gaussian_filter, shift as ndshift, uniform_filter
from heightmap.data.gamus import index_tiles
from heightmap.metrics import sharpness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--crop", type=int, default=518)
    a = ap.parse_args()

    ids = [t for _c, t in index_tiles(a.root, a.split)][:a.n]
    gts = []
    for tid in ids:
        with h5py.File(f"{a.root}/heights/{a.split}/{tid}_AGL.h5") as f:
            g = np.asarray(f["image"][()], np.float32)
        H = g.shape[0]; o = max(0, (H - a.crop) // 2)
        g = g[o:o+a.crop, o:o+a.crop]
        if np.isfinite(g).all():
            gts.append(g)
    print(f"{len(gts)} tiles, {a.crop}x{a.crop}\n")

    def score(fn, tag):
        v = []
        for g in gts:
            m = np.isfinite(g) & (g > -100.0)
            v.append(sharpness(fn(g), g, m)["ratio"])
        v = np.array(v, float)
        print(f"  {tag:<34} sharpness {np.nanmean(v):.3f}   (median {np.nanmedian(v):.3f})")

    print("degradations applied to the GROUND TRUTH, scored against itself:")
    score(lambda g: g.copy(), "identity (perfect)")
    for s in (0.25, 0.5, 1.0, 2.0):
        score(lambda g, s=s: ndshift(g, (s, 0), order=1, mode="nearest"),
              f"shifted {s} px")
    for s in (0.5, 1.0, 2.0, 3.0):
        score(lambda g, s=s: gaussian_filter(g, s), f"gaussian blur sigma={s}")
    for k in (3, 5, 9):
        score(lambda g, k=k: uniform_filter(g, k), f"box blur {k}x{k}")
    score(lambda g: g + np.random.default_rng(0).normal(0, 0.5, g.shape).astype(np.float32),
          "gaussian noise 0.5 m")
    score(lambda g: np.round(g), "rounded to 1 m")

    print("\nCompare against the measured model: 0.129-0.133 across every configuration\n"
          "tried (baseline, guided, flattened, refined, tiled).  Find which degradation\n"
          "row that sits between -- that is what the model's output is equivalent to.")


if __name__ == "__main__":
    main()
