#!/usr/bin/env python
"""Render a few GAMUS tiles as shareable PNGs: RGB | height (AGL) | land cover.

For showing the team what the training data actually is.  One PNG per tile, each with
the numbers that matter -- ground sample distance, tile footprint in metres, the height
distribution, and the building fraction -- so the picture is not just decorative.

Class ids come from heightmap/data/classes.py, which is the authority.  Building is 3,
not 2; getting that wrong is how building-only metrics ended up measuring
low-vegetation once already.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from heightmap.data.gamus import index_tiles, IMAGE_KINDS
from heightmap.data.classes import NAMES, BUILDING, N_CLASSES

# One colour per id 0..6, plus a distinct one for the 255 nodata label that only NYC uses.
CLS_COLOURS = ["#3b3b3b",   # 0 unlabelled
               "#c8b88a",   # 1 ground
               "#9ccb63",   # 2 low-vegetation
               "#e05a4e",   # 3 building
               "#4a90d9",   # 4 water
               "#8e8e8e",   # 5 road
               "#1f7a3d"]   # 6 tree
NODATA_COLOUR = "#ff00ff"   # 255, NYC only -- loud on purpose, it should be obvious


def load(root, split, tid):
    with h5py.File(f"{root}/heights/{split}/{tid}_AGL.h5") as f:
        agl = np.asarray(f["image"][()], np.float32)
    with h5py.File(f"{root}/classes/{split}/{tid}_CLS.h5") as f:
        cls = np.asarray(f["image"][()])
    img = None
    for kind in IMAGE_KINDS:                      # DC/PHL _RGB, NYC _IMG
        p = f"{root}/images/{split}/{tid}_{kind}.h5"
        if os.path.exists(p):
            with h5py.File(p) as f:
                img = np.asarray(f["image"][()])
            break
    return img, agl, cls


def render(tid, city, img, agl, cls, gsd, out_path, hi_pct=99.0):
    finite = np.isfinite(agl)
    a = np.where(finite, agl, np.nan)
    vals = a[finite]
    # Colour scale on a high percentile, not the max: one 390 m outlier would flatten
    # every building in the tile to the bottom of the ramp.
    vmax = float(np.percentile(vals, hi_pct)) if vals.size else 1.0
    vmax = max(vmax, 1.0)
    bld = float((cls == BUILDING).mean())
    n_nodata = int((cls == 255).sum())
    n_neg = int((vals < 0).sum())

    show = np.clip(np.where(np.isnan(a), 0.0, a), 0, vmax)
    cmap_c = ListedColormap(CLS_COLOURS + [NODATA_COLOUR])
    disp = np.where(cls == 255, N_CLASSES, np.clip(cls, 0, N_CLASSES - 1)).astype(int)

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.9))
    fig.patch.set_facecolor("white")

    ax[0].imshow(img)
    ax[0].set_title("RGB — the only model input", fontsize=12, fontweight="bold")

    im = ax[1].imshow(show, cmap="viridis", vmin=0, vmax=vmax)
    ax[1].set_title(f"Height above ground (nDSM/AGL)\nwhat we predict, in metres",
                    fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.set_label("metres above local ground", fontsize=9)

    ax[2].imshow(disp, cmap=cmap_c, norm=BoundaryNorm(np.arange(-0.5, N_CLASSES + 1), cmap_c.N))
    ax[2].set_title("Land cover (auxiliary supervision)", fontsize=12, fontweight="bold")
    handles = [Patch(facecolor=CLS_COLOURS[i], label=f"{i} {NAMES[i]}") for i in range(N_CLASSES)]
    if n_nodata:
        handles.append(Patch(facecolor=NODATA_COLOUR, label="255 nodata (NYC only)"))
    ax[2].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                 fontsize=8, frameon=False)

    for x in ax:
        x.set_xticks([]); x.set_yticks([])

    h, w = agl.shape
    p = np.percentile(vals, [50, 90, 99]) if vals.size else [0, 0, 0]
    sub = (f"GAMUS  ·  {city}  ·  {tid}  ·  {w}×{h} px at {gsd} m/px "
           f"= {w*gsd:.0f} × {h*gsd:.0f} m on the ground\n"
           f"height  p50 {p[0]:.1f} m   p90 {p[1]:.1f} m   p99 {p[2]:.1f} m   "
           f"max {np.nanmax(a):.1f} m   ·   building pixels {100*bld:.1f}%"
           f"   ·   colour scale clipped at p{hi_pct:g} = {vmax:.0f} m")
    if n_neg or n_nodata or (~finite).sum():
        sub += (f"\nquirks in this tile: {n_neg} negative height px, "
                f"{int((~finite).sum())} non-finite, {n_nodata} class-255 nodata")
    fig.suptitle(sub, fontsize=10.5, y=0.995, linespacing=1.45)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(top=0.80, wspace=0.06)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(tile=tid, city=city, building_frac=round(bld, 4),
                p50=round(float(p[0]), 2), p99=round(float(p[2]), 2),
                max=round(float(np.nanmax(a)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="train")
    ap.add_argument("--gsd", type=float, default=0.33)
    ap.add_argument("--out", default="outputs/samples")
    ap.add_argument("--per-city", type=int, default=1)
    ap.add_argument("--scan", type=int, default=40, help="tiles per city to scan when choosing")
    ap.add_argument("--min-building", type=float, default=0.12)
    ap.add_argument("--tiles", nargs="*", default=[], help="explicit tile ids, skips selection")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tiles = index_tiles(a.root, a.split)
    by_city = {}
    for city, tid in tiles:
        by_city.setdefault(city, []).append(tid)

    if a.tiles:
        chosen = [(t.split("_")[0], t) for t in a.tiles]
    else:
        chosen = []
        for city in sorted(by_city):
            # Spread the scan across the city instead of taking the first N, which would
            # sample one corner of one flight line.
            ids = by_city[city]
            step = max(1, len(ids) // a.scan)
            cand = []
            for tid in ids[::step][:a.scan]:
                try:
                    with h5py.File(f"{a.root}/classes/{a.split}/{tid}_CLS.h5") as f:
                        c = np.asarray(f["image"][()])
                    with h5py.File(f"{a.root}/heights/{a.split}/{tid}_AGL.h5") as f:
                        hgt = np.asarray(f["image"][()], np.float32)
                except Exception:
                    continue
                b = float((c == BUILDING).mean())
                fin = hgt[np.isfinite(hgt)]
                tall = float(np.percentile(fin, 99)) if fin.size else 0.0
                if b >= a.min_building:
                    cand.append((b * min(tall, 60.0), tid))   # buildings AND relief
            cand.sort(reverse=True)
            chosen += [(city, t) for _s, t in cand[:a.per_city]]

    meta = []
    for city, tid in chosen:
        img, agl, cls = load(a.root, a.split, tid)
        if img is None:
            print(f"skip {tid}: no image"); continue
        out = os.path.join(a.out, f"{tid}.png")
        meta.append(render(tid, city, img, agl, cls, a.gsd, out))
        print(f"wrote {out}")
    with open(os.path.join(a.out, "samples.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
