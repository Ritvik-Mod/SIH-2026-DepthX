#!/usr/bin/env python
"""Fetch swisstopo scenes: orthophoto + DSM + DTM over steep, built-up terrain.

    python scripts/download_swisstopo.py --n 10 --out data/swisstopo

WHY SWITZERLAND.  Every dataset this project has used so far is flat: GAMUS is DC / NYC
/ Philadelphia, US3D is Jacksonville and Omaha.  The problem statement grades height
accuracy across urban, sparse, HILLY and forested landscapes, and nothing in the repo
could measure the hilly case.  Switzerland is houses on inclines, and swisstopo publishes
all three layers we need over the same ground, free and without registration:

    SWISSIMAGE dop10          0.1 m orthophoto   -- the top view
    swissSURFACE3D Raster     0.5 m DSM          -- top of everything
    swissALTI3D               0.5 m DTM          -- bare earth

Having the DSM *and* the DTM is the point.  It gives:

  * AGL ground truth for free, as DSM - DTM, on terrain that actually slopes
  * the first real check of DSM = DTM + AGL, which this project has asserted
    everywhere and never once verified
  * real bare earth to tune dem_to_dtm()'s opening window against, instead of the
    guessed 120 m default

All three are Cloud-Optimised GeoTIFFs, so this reads only the window it needs over HTTP
rather than pulling 65 MB tiles and throwing most of them away.

VERTICAL DATUM.  swisstopo elevation is LN02 (EPSG:5728), NOT the EGM2008 this project
defaults to.  It is written into every sidecar here.  Mixing it with an EGM2008 source
without a geoid conversion is a silent multi-metre error -- exactly the trap documented
in SPEC_DSM.md.

Licence: swisstopo open government data, free to use with attribution. See the LICENSE
note written into data/swisstopo/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections"
IMAGE = "ch.swisstopo.swissimage-dop10"
DSM = "ch.swisstopo.swisssurface3d-raster"
DTM = "ch.swisstopo.swissalti3d"

LV95 = "EPSG:2056"                 # Swiss projected CRS, units are metres
VERTICAL_DATUM = "LN02 (EPSG:5728, Swiss national levelling network)"

#: Steep, built-up places.  More candidates than we need: each is scored against the real
#: data below and only the best survive, because "looks hilly on a map" is not a
#: measurement and a tile of bare forest is useless for a building-height benchmark.
CANDIDATES = [
    ("lausanne",     46.5210,  6.6320), ("lausanne_2",   46.5170,  6.6450),
    ("vevey",        46.4620,  6.8430), ("montreux",     46.4340,  6.9110),
    ("lugano",       46.0050,  8.9520), ("locarno",      46.1700,  8.7990),
    ("chur",         46.8520,  9.5300), ("sion",         46.2330,  7.3600),
    ("sierre",       46.2920,  7.5350), ("neuchatel",    46.9920,  6.9310),
    ("brienz",       46.7540,  8.0350), ("spiez",        46.6860,  7.6800),
    ("weggis",       47.0330,  8.4320), ("zermatt",      46.0210,  7.7480),
    ("stmoritz",     46.4980,  9.8380), ("wengen",       46.6080,  7.9220),
    ("bellinzona",   46.1940,  9.0240), ("thun",         46.7580,  7.6280),
    ("baden",        47.4730,  8.3080), ("fribourg",     46.8060,  7.1620),
    # swissSURFACE3D was acquired in phases and does not cover every canton yet, so the
    # list is padded well past --n: several of the above return no DSM at all.
    ("luzern",       47.0490,  8.3060), ("zug",          47.1680,  8.5150),
    ("rapperswil",   47.2260,  8.8180), ("schaffhausen", 47.6960,  8.6320),
    ("winterthur",   47.4990,  8.7240), ("baden_2",      47.4790,  8.3020),
    ("vitznau",      47.0090,  8.4840), ("beckenried",   46.9660,  8.4750),
    ("stgallen",     47.4250,  9.3770), ("biel",         47.1370,  7.2470),
]


def stac_asset(collection: str, lon: float, lat: float, want: str):
    """-> href of the asset whose name contains `want`, for the tile covering the point."""
    import urllib.request

    d = 0.002
    url = (f"{STAC}/{collection}/items"
           f"?bbox={lon-d:.5f},{lat-d:.5f},{lon+d:.5f},{lat+d:.5f}&limit=1")
    with urllib.request.urlopen(url, timeout=90) as r:
        feats = json.load(r).get("features", [])
    if not feats:
        return None
    for name, asset in feats[0]["assets"].items():
        if want in name and name.endswith(".tif"):
            return asset["href"]
    return None


def read_window(href, transform, shape, resampling):
    """Read `href` onto the given grid. COG + HTTP range requests: only this window moves."""
    import rasterio
    from rasterio.warp import reproject

    dst = np.zeros(shape, np.float32)
    with rasterio.open(href) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=LV95,
                  resampling=resampling,
                  src_nodata=src.nodata, dst_nodata=np.nan)
    return dst


def grid(lon, lat, size_px, gsd):
    """A north-up LV95 grid of size_px square at `gsd` m, centred on lon/lat."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", LV95, [lon], [lat])
    half = size_px * gsd / 2.0
    return from_origin(xs[0] - half, ys[0] + half, gsd, gsd)


def score(name, lon, lat, gsd, probe_px=256):
    """Is this location actually steep AND actually built up?  Measure, do not assume."""
    import rasterio
    from rasterio.warp import Resampling

    dsm_h = stac_asset(DSM, lon, lat, "_0.5_")
    dtm_h = stac_asset(DTM, lon, lat, "_0.5_")
    if not (dsm_h and dtm_h):
        return None

    tr = grid(lon, lat, probe_px, gsd * 4)          # coarse probe: cheap
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        dsm = read_window(dsm_h, tr, (probe_px, probe_px), Resampling.bilinear)
        dtm = read_window(dtm_h, tr, (probe_px, probe_px), Resampling.bilinear)

    ok = np.isfinite(dsm) & np.isfinite(dtm)
    if ok.mean() < 0.9:
        return None
    agl = np.where(ok, dsm - dtm, np.nan)

    gy, gx = np.gradient(np.where(ok, dtm, np.nan), gsd * 4)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

    return {
        "name": name, "lon": lon, "lat": lat,
        "relief_m": float(np.nanmax(dtm) - np.nanmin(dtm)),
        "mean_slope_deg": float(np.nanmean(slope_deg)),
        "built_fraction": float(np.nanmean(agl > 3.0)),
        "max_agl_m": float(np.nanmax(agl)),
    }


def export(sc, out_dir, size_px, gsd):
    import rasterio
    from rasterio.warp import Resampling

    name, lon, lat = sc["name"], sc["lon"], sc["lat"]
    tr = grid(lon, lat, size_px, gsd)
    shape = (size_px, size_px)

    img_h = stac_asset(IMAGE, lon, lat, "_0.1_")
    dsm_h = stac_asset(DSM, lon, lat, "_0.5_")
    dtm_h = stac_asset(DTM, lon, lat, "_0.5_")
    if not (img_h and dsm_h and dtm_h):
        print(f"  {name}: missing a product, skipped")
        return None

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        # cubic for imagery (0.1 -> 0.33 m is a downsample), bilinear for elevation:
        # elevation is a smooth field and cubic can overshoot at building walls.
        from rasterio.warp import reproject
        rgb = np.zeros((3,) + shape, np.float32)
        with rasterio.open(img_h) as src:
            for i in range(3):
                reproject(source=rasterio.band(src, i + 1), destination=rgb[i],
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=tr, dst_crs=LV95,
                          resampling=Resampling.cubic)
        dsm = read_window(dsm_h, tr, shape, Resampling.bilinear)
        dtm = read_window(dtm_h, tr, shape, Resampling.bilinear)

    valid = np.isfinite(dsm) & np.isfinite(dtm)
    raw = dsm - dtm

    # AGL is clipped at 0 because nothing stands below the ground, and because GAMUS's
    # AGL is >= 0 too -- a label that goes negative would not be comparable.
    #
    # The clip is the ONLY reason DTM + AGL does not reproduce DSM exactly. swissALTI3D
    # and swissSURFACE3D are independent products from separate acquisitions, so their
    # difference carries decimetre-scale noise; on flat ground its median is within a few
    # centimetres of zero, which is the real validation of DSM = DTM + AGL. The residual
    # is measured and recorded rather than hidden, so nobody mistakes the clip for a bug.
    agl = np.where(valid, np.maximum(raw, 0.0), -9999.0).astype(np.float32)

    # How well do the two independent products agree where nothing is standing?
    #
    # Selecting "ground" by DTM slope does NOT work: the DTM is smooth UNDERNEATH
    # buildings, so a flat roof on flat terrain passes a slope test and the statistic
    # then measures rooftops. In a dense scene that reads several metres and looks like
    # a product disagreement when it is just a building. The 25th percentile of the raw
    # difference is used instead -- most of a scene is open ground, so a low quantile
    # lands on it without needing to classify anything.
    q25 = float(np.percentile(raw[valid], 25))
    consistency = {
        "identity": "DTM + AGL == DSM wherever DSM >= DTM; elsewhere AGL is clipped to 0",
        "ground_agreement_p25_m": round(q25, 4),
        "ground_agreement_note": "25th percentile of DSM-DTM; a robust stand-in for open "
                                 "ground. Near 0 means the two products agree there.",
        "raw_diff_negative_fraction": round(float(np.mean(raw[valid] < 0)), 4),
        "raw_diff_below_half_metre_fraction": round(float(np.mean(raw[valid] < -0.5)), 4),
        "max_clip_residual_m": round(float(np.abs(np.minimum(raw[valid], 0.0)).max()), 4),
    }

    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    base = dict(driver="GTiff", height=size_px, width=size_px, crs=LV95, transform=tr,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)

    with rasterio.open(d / f"{name}_RGB.tif", "w", count=3, dtype="uint8", **base) as o:
        o.write(np.clip(rgb, 0, 255).astype(np.uint8))
        o.update_tags(gsd_m=str(gsd), source="SWISSIMAGE dop10 (swisstopo)")

    for kind, arr in (("DSM", dsm), ("DTM", dtm), ("AGL", agl)):
        a = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
        with rasterio.open(d / f"{name}_{kind}.tif", "w", count=1, dtype="float32",
                           nodata=-9999.0, **base) as o:
            o.write(a, 1)
            o.update_tags(quantity=kind, units="metres",
                          vertical_datum=VERTICAL_DATUM if kind != "AGL" else "n/a (AGL)",
                          source="swisstopo")

    meta = {
        "name": name, "lat": lat, "lon": lon,
        "crs": LV95, "gsd_m": gsd, "size_px": size_px,
        "footprint_m": round(size_px * gsd, 1),
        "vertical_datum": VERTICAL_DATUM,
        "products": {"RGB": "SWISSIMAGE dop10 0.1 m", "DSM": "swissSURFACE3D Raster 0.5 m",
                     "DTM": "swissALTI3D 0.5 m", "AGL": "derived: DSM - DTM, clipped at 0"},
        "terrain": {k: round(sc[k], 2) for k in
                    ("relief_m", "mean_slope_deg", "built_fraction", "max_agl_m")},
        "agl_consistency": consistency,
        "licence": "swisstopo open government data; attribution required",
        "note": "AGL is DSM - DTM from LiDAR, so it is ground truth, not a prediction.",
    }
    (d / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="how many scenes to keep")
    ap.add_argument("--out", default="data/swisstopo")
    ap.add_argument("--size", type=int, default=1024, help="pixels square")
    ap.add_argument("--gsd", type=float, default=0.33,
                    help="metres/pixel; 0.33 matches the model's training GSD")
    a = ap.parse_args(argv)

    print(f"scoring {len(CANDIDATES)} candidate locations for slope and buildings...")
    scored = []
    for name, lat, lon in CANDIDATES:
        try:
            s = score(name, lon, lat, a.gsd)
        except Exception as e:
            print(f"  {name:12} failed ({type(e).__name__})")
            continue
        if s is None:
            print(f"  {name:12} no data")
            continue
        scored.append(s)
        print(f"  {name:12} relief {s['relief_m']:6.1f} m   slope {s['mean_slope_deg']:5.1f} deg"
              f"   built {s['built_fraction']*100:5.1f}%")

    # Want both: real terrain relief AND real buildings. A bare alp scores high on slope
    # and is useless here; a flat city centre is what we already have in GAMUS.
    usable = [s for s in scored if s["built_fraction"] > 0.04 and s["mean_slope_deg"] > 4.0]
    usable.sort(key=lambda s: s["mean_slope_deg"] * min(s["built_fraction"], 0.5), reverse=True)
    keep = usable[:a.n]

    print(f"\n{len(usable)} of {len(scored)} are both steep and built up; keeping {len(keep)}")
    out = []
    for s in keep:
        print(f"  exporting {s['name']} ...", flush=True)
        m = export(s, a.out, a.size, a.gsd)
        if m:
            out.append(m)

    idx = Path(a.out) / "index.json"
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {len(out)} scenes to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
