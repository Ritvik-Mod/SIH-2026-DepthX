#!/usr/bin/env python
"""Wrap an ordinary image in a real GeoTIFF so the georeferenced path can run.

    python scripts/make_geotiff.py --image delhi.png --lat 28.6139 --lon 77.2090 \
                                   --gsd 0.33 --out delhi_geo.tif

Accepts PNG / JPG, a plain TIFF, or a GAMUS *_RGB.h5 / *_IMG.h5 tile.

WHY THIS EXISTS.  Two things in this pipeline need spatial metadata that a plain photo
does not carry: the GSD (metres per pixel, without which no output can be metric) and
the location (without which no DEM can be fetched and no output can be placed on a map).
A GeoTIFF carries both.  This script attaches them.

WHAT IT DOES NOT DO.  It does not discover where your image was taken -- you tell it,
and it believes you.  The coordinates written are exactly as accurate as the --lat/--lon
you supply.  That is fine for placing a scene on a map and for fetching a DEM, and it is
NOT survey-grade georeferencing.  The output records this in its tags as
georeferencing="declared by operator, not surveyed" so nobody downstream mistakes a
typed coordinate for a measured one.

The image is assumed to be north-up and unrotated, which is true of orthophotos and of
anything screenshotted from a map.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np


def utm_epsg(lat: float, lon: float) -> int:
    """The UTM zone containing this point, as an EPSG code.

    UTM is a metric projection: one unit is one metre, which is what makes a GSD in
    metres meaningful.  Writing lat/lon (EPSG:4326) instead would make the pixel size a
    fraction of a degree, and degrees are not metres -- a 0.33 m pixel would be recorded
    as 0.33 DEGREES, roughly 37 km, and every downstream size would be wrong by 10^5.
    """
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def read_any(path):
    """-> HxWx3 uint8, from PNG/JPG, a TIFF, or a GAMUS .h5 tile."""
    low = str(path).lower()

    if low.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = next(iter(f.keys()))
            a = np.asarray(f[key])
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
            a = a.transpose(1, 2, 0)                     # CHW -> HWC
        return _to_uint8(a[..., :3])

    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(str(path)) as ds:
            a = ds.read()
        if a.shape[0] < 3:
            a = np.repeat(a[:1], 3, 0)
        return _to_uint8(a[:3].transpose(1, 2, 0))

    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def _to_uint8(a):
    a = np.asarray(a)
    if a.dtype == np.uint8:
        return a
    f = a.astype(np.float32)
    lo, hi = np.percentile(f, 1), np.percentile(f, 99)   # 11/16-bit -> 8-bit
    return (np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="PNG / JPG / TIFF / GAMUS .h5")
    ap.add_argument("--out", required=True, help="output .tif")
    ap.add_argument("--lat", type=float, required=True, help="latitude of the image CENTRE")
    ap.add_argument("--lon", type=float, required=True, help="longitude of the image CENTRE")
    ap.add_argument("--gsd", type=float, default=0.33,
                    help="metres per pixel (default 0.33, the model's training GSD)")
    ap.add_argument("--epsg", type=int, default=None,
                    help="override the auto-chosen UTM zone")
    a = ap.parse_args(argv)

    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform

    rgb = read_any(a.image)
    h, w = rgb.shape[:2]

    # The model conditions on log(GSD) and was trained over 0.165-0.65 m/px.  Outside
    # that band it is extrapolating and the metres it returns are not trustworthy.
    if not (0.165 <= a.gsd <= 0.65):
        print(f"  WARNING: gsd {a.gsd} m/px is outside the model's trained band "
              f"(0.165-0.65). Heights from it will not be reliable.")

    epsg = a.epsg or utm_epsg(a.lat, a.lon)
    xs, ys = warp_transform("EPSG:4326", f"EPSG:{epsg}", [a.lon], [a.lat])
    cx, cy = xs[0], ys[0]

    # from_origin takes the TOP-LEFT corner, so step half the image out from the centre.
    west = cx - (w / 2.0) * a.gsd
    north = cy + (h / 2.0) * a.gsd
    transform = from_origin(west, north, a.gsd, a.gsd)

    prof = dict(driver="GTiff", height=h, width=w, count=3, dtype="uint8",
                crs=f"EPSG:{epsg}", transform=transform,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(a.out, "w", **prof) as ds:
        ds.write(rgb.transpose(2, 0, 1))
        ds.update_tags(
            gsd_m=str(a.gsd),
            centre_lat=str(a.lat), centre_lon=str(a.lon),
            georeferencing="declared by operator, not surveyed",
            source_image=os.path.basename(a.image),
        )

    print(f"wrote {a.out}")
    print(f"  {w} x {h} px  @ {a.gsd} m/px   =  {w*a.gsd:.0f} x {h*a.gsd:.0f} m on the ground")
    print(f"  EPSG:{epsg} (UTM zone {epsg % 100}{'N' if epsg < 32700 else 'S'})")
    print(f"  centre {a.lat}, {a.lon}")
    print("\n  Coordinates are the ones you typed, not surveyed ones -- recorded in the")
    print("  file's tags so nobody downstream mistakes them for measured positions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
