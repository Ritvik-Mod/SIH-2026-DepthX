#!/usr/bin/env python
"""Shrink the GAMUS download in place, losslessly where it matters.

As published it is 80 GB, which is a lot to ask of a shared-cluster quota:
    heights   36.6 GB   float32   -> float16   18.3 GB
    classes   16.0 GB   float32   -> uint8      4.0 GB   (values are ids 0..6)
    images    27.5 GB   uint8     -> unchanged 27.5 GB
                                     total     ~50 GB

Class ids 0..6 in uint8 is exactly lossless.  Heights in float16 lose ~3 cm of precision
at 40 m and ~6 cm at 100 m -- two orders of magnitude below the ~1.5 m MAE the model is
aiming for, so it costs nothing real.  Everything stays one-file-per-tile with the same
keys, so the dataloader needs no change (it already casts on read).

Run with --check first: it reports the saving and verifies round-trip fidelity on a
sample before touching anything.
"""
import argparse, os, shutil, sys
from pathlib import Path
import numpy as np
import h5py

KIND = {"heights": ("AGL", np.float16), "classes": ("CLS", np.uint8), "images": ("RGB", None)}


def tile_files(root: Path):
    for sub, (kind, dt) in KIND.items():
        for split in ("train", "val", "test"):
            d = root / sub / split
            if d.is_dir():
                for f in sorted(d.glob(f"*_{kind}.h5")):
                    yield f, dt


def repack_one(f: Path, dt, compress: str | None) -> tuple[int, int]:
    before = f.stat().st_size
    with h5py.File(f, "r") as h:
        a = np.asarray(h["image"][()])
    if dt is None and not compress:
        return before, before
    out = a if dt is None else a.astype(dt)
    if dt is not None and np.issubdtype(dt, np.integer):
        lo, hi = np.iinfo(dt).min, np.iinfo(dt).max
        if a.min() < lo or a.max() > hi:
            raise ValueError(f"{f.name}: values [{a.min()},{a.max()}] do not fit {dt}")
    tmp = f.with_suffix(".h5.tmp")
    kw = dict(compression=compress) if compress else {}
    with h5py.File(tmp, "w") as h:
        h.create_dataset("image", data=out, **kw)
    os.replace(tmp, f)                        # atomic
    return before, f.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--check", action="store_true", help="verify fidelity on a sample, change nothing")
    ap.add_argument("--compress", default=None, choices=[None, "lzf", "gzip"],
                    help="optional; lzf is fast to read, gzip smaller but slower")
    a = ap.parse_args()
    root = Path(a.root)
    files = list(tile_files(root))
    if not files:
        sys.exit(f"no GAMUS tiles under {root}")

    if a.check:
        print(f"{len(files)} tiles found. verifying round-trip on a sample...")
        import random
        for f, dt in random.sample(files, min(12, len(files))):
            with h5py.File(f, "r") as h:
                a0 = np.asarray(h["image"][()])
            if dt is None:
                print(f"  {f.name:28s} {a0.dtype} unchanged")
                continue
            back = a0.astype(dt).astype(np.float32)
            err = float(np.abs(back - a0.astype(np.float32)).max())
            ok = "OK" if (dt != np.uint8 or err == 0) else "LOSSY"
            print(f"  {f.name:28s} {a0.dtype}->{np.dtype(dt).name}  max abs err {err:.5f}  {ok}")
        tot = sum(f.stat().st_size for f, _ in files)
        print(f"\ncurrent total {tot/1e9:.1f} GB; expected after repack ~{tot*0.63/1e9:.0f} GB")
        return

    before_tot = after_tot = 0
    for i, (f, dt) in enumerate(files, 1):
        b, aft = repack_one(f, dt, a.compress)
        before_tot += b; after_tot += aft
        if i % 500 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {before_tot/1e9:.1f} -> {after_tot/1e9:.1f} GB", flush=True)
    print(f"\ndone: {before_tot/1e9:.1f} GB -> {after_tot/1e9:.1f} GB "
          f"(saved {(before_tot-after_tot)/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
