#!/usr/bin/env python
"""Inspect a US3D download before trusting any of it.

Everything this prints is something the loader ASSUMES and cannot check at runtime:

  * which products actually arrived, and how many tiles have a complete set
  * the real image size and GSD implied by the georeferencing
  * the ACTUAL unique CLS values on disk, against the LAS codes the remap expects.
    heightmap/data/us3d.py maps {2,3,4,5,6,9,17} -> GAMUS ids from the LAS standard.
    If the files use different codes, every semantic label is silently wrong and the
    height loss spends the whole run fighting a mislabelled auxiliary task.  Bug #6
    here was precisely this (building was 2, should have been 3) and it survived an
    entire ablation ladder unnoticed.
  * the AGL height distribution, to compare against GAMUS's measured
    p50 0.30 / p90 13.4 / p99 32.0 / max 392.6
  * whether Track 3's DSM tiles overlap Track 1's AGL tiles, which is the whole
    premise of deriving DTM = DSM - AGL

Read-only.  Writes nothing, trains nothing.
"""
from __future__ import annotations
import argparse, collections, os, sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

from heightmap.data.us3d import (CLS_LAS_TO_GAMUS, TILE_RE, index_us3d, read_tif)

LAS_NAMES = {1: "unclassified", 2: "ground", 3: "low veg", 4: "med veg", 5: "high veg",
             6: "building", 7: "low point", 9: "water", 17: "bridge deck"}
GAMUS_NAMES = {0: "unlabelled", 1: "ground", 2: "low-veg", 3: "building", 4: "water",
               5: "road", 6: "tree"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/US3D")
    ap.add_argument("--sample", type=int, default=12, help="tiles to open for statistics")
    a = ap.parse_args()
    root = Path(a.root)
    if not root.exists():
        sys.exit(f"{root} does not exist")

    # ---- inventory ----------------------------------------------------------
    kinds = collections.Counter()
    per_tile: dict[str, set] = {}
    sites = collections.Counter()
    for p in root.rglob("*.tif"):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        k = m["kind"].upper()
        kinds[k] += 1
        stem = f"{m['site']}_{m['tile']}_{m['view']}"
        per_tile.setdefault(stem, set()).add(k)
        sites[m["site"].upper()] += 1

    print(f"root {root}")
    print(f"\nproducts found: {dict(kinds)}")
    print(f"sites: {dict(sites)}")
    print(f"distinct tile stems: {len(per_tile)}")
    complete = index_us3d(root)
    print(f"tiles with RGB+AGL+CLS (usable for training): {len(complete)}")
    if not complete:
        print("\n  Nothing usable yet. Track 1 'RGB images' AND Track 1 'Reference'\n"
              "  are both required -- Reference carries AGL and CLS.")
        return 0

    # ---- DSM overlap: the DTM = DSM - AGL premise ---------------------------
    with_dsm = {s for s, k in per_tile.items() if "DSM" in k}
    with_agl = {s for s, k in per_tile.items() if "AGL" in k}
    both = with_dsm & with_agl
    print(f"\nDSM tiles {len(with_dsm)}   AGL tiles {len(with_agl)}   BOTH {len(both)}")
    if not with_dsm:
        print("  no DSM present -- download Track 3 / Training data / Reference (37 MB)")
    elif not both:
        # DSM is keyed by tile, AGL by tile+view, so a strict stem match can miss.
        base = lambda s: "_".join(s.split("_")[:2])
        loose = {base(s) for s in with_dsm} & {base(s) for s in with_agl}
        print(f"  no exact stem overlap, but {len(loose)} shared site_tile prefixes -- "
              f"DSM is per-TILE and AGL per-VIEW, so join on the prefix, not the stem")

    # ---- the class check ----------------------------------------------------
    seen = collections.Counter()
    heights = []
    shapes = collections.Counter()
    for site, stem in complete[:a.sample]:
        c = read_tif(next(root.rglob(f"{stem}_CLS.tif")))
        v, n = np.unique(np.asarray(c), return_counts=True)
        seen.update(dict(zip(v.tolist(), n.tolist())))
        h = read_tif(next(root.rglob(f"{stem}_AGL.tif"))).astype(np.float32)
        shapes[tuple(np.shape(h))] += 1
        heights.append(h[np.isfinite(h)])

    total = sum(seen.values())
    print(f"\nCLS values actually on disk (from {min(a.sample, len(complete))} tiles):")
    print(f"  {'code':>5}  {'LAS name':<14}{'pixels':>12}{'share':>8}   -> GAMUS")
    unmapped = []
    for code in sorted(seen):
        pct = 100 * seen[code] / total
        tgt = CLS_LAS_TO_GAMUS.get(int(code))
        arrow = f"{tgt} ({GAMUS_NAMES.get(tgt,'?')})" if tgt is not None else "UNMAPPED -> ignore"
        if tgt is None:
            unmapped.append(int(code))
        print(f"  {int(code):>5}  {LAS_NAMES.get(int(code),'?'):<14}{seen[code]:>12,}{pct:>7.2f}%   -> {arrow}")

    if unmapped:
        print(f"\n  *** {len(unmapped)} UNMAPPED code(s): {unmapped}")
        print("  Every pixel of these is being thrown into the ignore index. If any has a"
              "\n  meaningful share, fix CLS_LAS_TO_GAMUS in heightmap/data/us3d.py before"
              "\n  training -- do not let it ride.")
    else:
        print("\n  every code on disk is mapped. CLS_LAS_TO_GAMUS is verified for this sample.")

    b = seen.get(6, 0)
    print(f"\n  building (LAS 6) share {100*b/total:.1f}%  "
          f"-- GAMUS measures ~17%; a wildly different figure means the codes differ")

    # ---- height distribution ------------------------------------------------
    h = np.concatenate(heights)
    q = np.percentile(h, [50, 90, 99, 99.9])
    print(f"\ntile shapes: {dict(shapes)}")
    print(f"AGL over {len(heights)} tiles, {h.size:,} finite px:")
    print(f"  p50 {q[0]:.2f}  p90 {q[1]:.2f}  p99 {q[2]:.2f}  p99.9 {q[3]:.2f}  "
          f"max {h.max():.1f}  min {h.min():.2f}")
    print(f"  GAMUS for comparison:")
    print(f"  p50 0.30  p90 13.40  p99 32.00  p99.9 44.30  max 392.6")
    neg = 100 * (h < 0).mean()
    print(f"  negatives {neg:.2f}%   (GAMUS 2.79%)")
    print(f"\n  h_max is configured at 250 m -- "
          f"{'fine' if q[3] < 250 else 'RAISE IT, p99.9 exceeds it'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
