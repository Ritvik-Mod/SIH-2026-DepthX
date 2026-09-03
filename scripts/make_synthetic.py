#!/usr/bin/env python
"""Generate GAMUS-shaped synthetic tiles so the whole pipeline is testable before the
real download finishes -- and so the 3D/calibration teams have a file to build against
on day one.  Rooftops are flat prisms over a smooth terrain-free ground, which is what
AGL looks like, with shadows cast consistently with a chosen sun angle."""
import argparse, json, math
from pathlib import Path
import numpy as np, h5py


def make_tile(rng, n=1024, sun_elev=45.0, sun_azim=140.0):
    agl = np.zeros((n, n), np.float32)
    rgb = np.zeros((n, n, 3), np.float32)
    rgb[:] = np.array([0.42, 0.42, 0.40]) + rng.normal(0, 0.02, (n, n, 1))
    cls = np.zeros((n, n), np.float32)          # 0 ground
    cls[:] = 1.0                                 # 1 = ground
    for _ in range(rng.integers(25, 55)):
        h = float(np.clip(rng.gamma(2.2, 5.0), 2, 90))
        bw, bh = rng.integers(40, 140), rng.integers(40, 140)
        y, x = rng.integers(0, n - bh), rng.integers(0, n - bw)
        agl[y:y + bh, x:x + bw] = h
        cls[y:y + bh, x:x + bw] = 3.0            # 3 = building
        rgb[y:y + bh, x:x + bw] = np.clip(rng.uniform(0.32, 0.62, 3), 0, 1)
    for _ in range(rng.integers(15, 40)):        # trees
        h = float(np.clip(rng.normal(9, 3), 2, 22)); r = int(rng.integers(8, 22))
        cy, cx = rng.integers(r, n - r), rng.integers(r, n - r)
        yy, xx = np.ogrid[-r:r, -r:r]
        d = (yy ** 2 + xx ** 2) <= r * r
        agl[cy - r:cy + r, cx - r:cx + r][d] = h
        cls[cy - r:cy + r, cx - r:cx + r][d] = 6.0   # 6 = tree
        rgb[cy - r:cy + r, cx - r:cx + r][d] = np.array([0.16, 0.34, 0.14])

    gsd = 0.5
    az, el = math.radians(sun_azim), math.radians(sun_elev)
    ux, uy = math.sin(az), -math.cos(az)
    occ = np.full((n, n), -1e9, np.float32)
    ys, xs = np.mgrid[0:n, 0:n]
    for t in [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 55, 75, 100, 140]:
        sy = np.clip((ys + t * uy).astype(int), 0, n - 1)
        sx = np.clip((xs + t * ux).astype(int), 0, n - 1)
        occ = np.maximum(occ, agl[sy, sx] - agl - t * gsd * math.tan(el))
    shadow = occ > 0
    # real shadows are darker AND bluer (sky-lit rather than sun-lit)
    rgb[shadow] *= np.array([0.38, 0.42, 0.52], np.float32)
    rgb = np.clip(rgb + rng.normal(0, 0.012, rgb.shape), 0, 1)
    return (rgb * 255).astype(np.uint8), agl, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS_synthetic")
    ap.add_argument("--per-split", type=int, default=8)
    ap.add_argument("--size", type=int, default=1024)
    a = ap.parse_args()
    root = Path(a.root); rng = np.random.default_rng(7); sun = {}; gidx = 0
    for split, cities in (("train", ["DC", "PHL", "NYC"]), ("val", ["DC", "PHL"]), ("test", ["DC", "NYC"])):
        for k in range(a.per_split):
            city = cities[k % len(cities)]
            tid = f"{city}_{gidx:03d}_{k:02d}"   # globally unique: ids must not collide across splits
            gidx += 1
            elev = float(rng.uniform(25, 60)); azim = float(rng.uniform(90, 250))
            rgb, agl, cls = make_tile(rng, a.size, elev, azim)
            for sub, kind, arr in (("images", "RGB", rgb), ("heights", "AGL", agl), ("classes", "CLS", cls)):
                d = root / sub / split; d.mkdir(parents=True, exist_ok=True)
                with h5py.File(d / f"{tid}_{kind}.h5", "w") as f:
                    f.create_dataset("image", data=arr)
            sun[tid] = [elev, azim]
    (root / "sun_angles.json").write_text(json.dumps(sun, indent=1))
    print(f"wrote {a.per_split*3} tiles per split under {root}, sun angles in sun_angles.json")


if __name__ == "__main__":
    main()
