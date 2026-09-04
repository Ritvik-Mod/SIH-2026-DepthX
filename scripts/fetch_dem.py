#!/usr/bin/env python
"""Fetch the ground under a georeferenced image: Copernicus GLO-30 -> a local DEM.

    python scripts/fetch_dem.py --image scene.tif --out dem.tif
    python -m heightmap.predict --ckpt ... --image scene.tif --dem dem.tif   # -> DSM

The PS names SRTM as the supplementary DEM.  GLO-30 is used instead: same 30 m posting,
same free access, no login, and it is the newer product (2011-2015 radar vs SRTM's
single 2000 campaign) with SRTM's voids already filled.  Anything the PS says about
SRTM applies unchanged -- both are 30 m global DEMs, and both are SURFACE models, which
is why heightmap/terrain.py strips structures out before using one as ground.

Only the window covering the image is read.  The tiles are cloud-optimised GeoTIFFs, so
GDAL fetches the few hundred kB of overlapping blocks over HTTP rather than the ~100 MB
tile.  That is the difference between this taking two seconds and taking ten minutes on
a shared connection.

Offline clusters: run this on a machine with internet, copy the small output across.
--urls prints exactly what would be fetched without fetching anything.
"""
import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
PAD_DEG = 0.02          # ~2 km.  The opening kernel needs DEM beyond the image edge,
                        # or the ground under the border is opened against nothing.


def tile_name(lat: int, lon: int) -> str:
    ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
    ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"


def tiles_for(w, s, e, n):
    """The 1x1 degree tiles covering a bounding box, named by their SW corner."""
    out = []
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            out.append((lat, lon, f"{BASE}/{tile_name(lat, lon)}/{tile_name(lat, lon)}.tif"))
    return out


def image_bounds_wgs84(path):
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.transform.is_identity:
            sys.exit(f"{path} is not georeferenced, so there is no footprint to fetch a "
                     "DEM for. A PNG/JPG can only produce a relative (rDSM) output.")
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds), ds.crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="georeferenced image whose footprint to cover")
    ap.add_argument("--bounds", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    help="lon/lat bounds instead of --image")
    ap.add_argument("--out", default="dem.tif")
    ap.add_argument("--pad", type=float, default=PAD_DEG,
                    help="degrees of margin beyond the footprint (default ~2 km)")
    ap.add_argument("--local-dir", default=None,
                    help="read tiles from this directory instead of the network")
    ap.add_argument("--urls", action="store_true", help="print the tile URLs and exit")
    a = ap.parse_args()

    if a.image:
        (w, s, e, n), crs = image_bounds_wgs84(a.image)
        print(f"image footprint  {w:.4f},{s:.4f} .. {e:.4f},{n:.4f}  (from {crs})")
    elif a.bounds:
        w, s, e, n = a.bounds
    else:
        sys.exit("pass --image or --bounds")
    w, s, e, n = w - a.pad, s - a.pad, e + a.pad, n + a.pad

    tl = tiles_for(w, s, e, n)
    print(f"{len(tl)} GLO-30 tile(s) cover it (padded {a.pad} deg):")
    for lat, lon, url in tl:
        src = os.path.join(a.local_dir, os.path.basename(url)) if a.local_dir else url
        print(f"  {tile_name(lat, lon)}   {src}")
    if a.urls:
        return

    import rasterio
    from rasterio.merge import merge
    from rasterio.windows import from_bounds
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")   # no directory
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")    # listing over HTTP

    opened, missing = [], []
    for lat, lon, url in tl:
        path = (os.path.join(a.local_dir, os.path.basename(url)) if a.local_dir
                else f"/vsicurl/{url}")
        try:
            opened.append(rasterio.open(path))
        except Exception as ex:
            # a tile that is entirely ocean is simply not published; that is not an error
            missing.append((tile_name(lat, lon), str(ex).splitlines()[0][:80]))
    if not opened:
        sys.exit("no DEM tiles could be opened:\n  " +
                 "\n  ".join(f"{t}: {m}" for t, m in missing) +
                 "\n(no internet? use --local-dir, or --urls to see what to download)")
    for t, m in missing:
        print(f"  note: {t} unavailable ({m}) -- ocean tiles are not published")

    arr, tr = merge(opened, bounds=(w, s, e, n))
    prof = opened[0].profile
    for ds in opened:
        ds.close()
    prof.update(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=1,
                transform=tr, compress="deflate", tiled=True)
    with rasterio.open(a.out, "w", **prof) as dst:
        dst.write(arr[0], 1)
        dst.update_tags(vertical_datum="EGM2008",
                        source="Copernicus GLO-30 (ESA), via AWS Open Data",
                        note="SURFACE model: includes buildings and vegetation. Strip "
                             "structures before using as ground (heightmap.terrain).")
    v = arr[0][np.isfinite(arr[0]) & (arr[0] > -1000)]
    print(f"\nwrote {a.out}  {arr.shape[2]}x{arr.shape[1]} posts")
    if v.size:
        print(f"  elevation {v.min():.0f} to {v.max():.0f} m  (EGM2008 orthometric)")
    print("  this is a SURFACE model -- predict.py's --terrain-mode opened strips the "
          "buildings out of it before adding ours.")


if __name__ == "__main__":
    main()
