#!/usr/bin/env python
"""Emit a contract-compliant sample output so the calibration, TDA and 3D stages can
build against a real file on day one, before the model exists."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from heightmap.export import write_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/contract_sample")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--gsd", type=float, default=0.33)  # GAMUS paper sec.1
    a = ap.parse_args()

    rng = np.random.default_rng(3)
    n = a.size
    agl = np.zeros((n, n), np.float32)
    for _ in range(60):                                   # buildings: flat prisms
        h = float(np.clip(rng.gamma(2.2, 5.0), 2, 80))
        bw, bh = rng.integers(30, 120), rng.integers(30, 120)
        y, x = rng.integers(0, n - bh), rng.integers(0, n - bw)
        agl[y:y + bh, x:x + bw] = h
    for _ in range(40):                                   # trees: rounded blobs
        h = float(np.clip(rng.normal(9, 3), 2, 22)); r = int(rng.integers(6, 18))
        cy, cx = rng.integers(r, n - r), rng.integers(r, n - r)
        yy, xx = np.ogrid[-r:r, -r:r]
        agl[cy - r:cy + r, cx - r:cx + r][(yy ** 2 + xx ** 2) <= r * r] = h
    sigma = (0.6 + 0.05 * agl + rng.random((n, n)) * 0.4).astype(np.float32)
    agl[:24, :24] = np.nan                                 # a nodata patch to exercise masking

    meta = write_outputs(a.out, "contract_sample", agl, sigma=sigma, gsd=a.gsd,
                         model_info={"model": "SAMPLE — synthetic, not a real prediction",
                                     "checkpoint": None},
                         infer_meta={"tile": 518, "overlap": 0.25, "tta": False,
                                     "levelled": True, "n_tiles": 0},
                         sun={"elevation_deg": 48.2, "azimuth_deg": 152.7, "source": "synthetic"},
                         assumptions=["Synthetic sample for interface development only."],
                         limitations=["Contains no real height information."])
    print(f"wrote {a.out}/contract_sample.{{tif,json,_preview.png}}")
    print(f"  quantity={meta['quantity']}  nodata={meta['nodata']}  bands={list(meta['bands'].values())}")
    print(f"  value range p1={meta['value_range']['p1']:.2f} p99={meta['value_range']['p99']:.2f}")
    print("\nRemember: band 1 is AGL. DSM = DTM + agl_metres.  ADD terrain, do not subtract.")


if __name__ == "__main__":
    main()
