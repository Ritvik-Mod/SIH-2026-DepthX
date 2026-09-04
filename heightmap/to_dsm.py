"""CLI: an AGL raster in, a DSM out.

    # simplest thing that works -- no downloads, no extra data
    python -m heightmap.to_dsm --agl outputs/contract_sample/contract_sample.tif \
                               --out outputs/dsm --base-elevation 12

    # no spatial metadata anywhere -> relative surface, honestly labelled
    python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm --mode relative

    # you have a DEM file for the area
    python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm --dem cop30.tif

    # you have surveyed points:  a CSV of  x,y,elevation
    python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm --gcp points.csv

Writes {stem}_dsm.tif, {stem}_dsm.json and a preview PNG.  See SPEC_DSM.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .dsm.convert import (DEFAULT_VERTICAL_DATUM, read_agl, to_rdsm, to_dsm, write_dsm)
from .dsm.terrain import constant_plane, gcp_plane, load_dem_as_dtm
from .dsm.calibrate import calibrate_to_flat_ground


def read_gcps(path):
    """CSV of x,y,elevation -- one surveyed point per line, header optional."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace("\t", ",").split(",")]
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except (ValueError, IndexError):
            continue                                    # header or junk line
    if not rows:
        raise SystemExit(f"{path}: no usable rows. Expected lines of  x,y,elevation")
    a = np.asarray(rows, float)
    return a[:, 0], a[:, 1], a[:, 2]


def read_semantic(path, shape):
    import rasterio
    with rasterio.open(str(path)) as ds:
        sem = ds.read(1)
    if sem.shape != shape:
        raise SystemExit(f"semantic map is {sem.shape}, AGL is {shape}; they must match")
    return sem


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Turn an AGL height raster into a DSM (elevation of the top of "
                    "everything). DSM = DTM + AGL -- terrain is ADDED.")
    ap.add_argument("--agl", required=True,
                    help="the .tif written by the height stage (band 1 = agl_metres)")
    ap.add_argument("--out", default="outputs/dsm")
    ap.add_argument("--mode", choices=("auto", "absolute", "relative"), default="auto",
                    help="auto: absolute metres if the input is georeferenced or a base "
                         "elevation is given, otherwise a relative surface")

    t = ap.add_argument_group("terrain (pick at most one; most trustworthy first)")
    t.add_argument("--gcp", metavar="CSV",
                   help="surveyed points, x,y,elevation per line -- most accurate")
    t.add_argument("--dem", metavar="TIF",
                   help="a DEM file; buildings are stripped from it first")
    t.add_argument("--base-elevation", type=float, metavar="M",
                   help="one ground elevation for the whole scene, metres above sea "
                        "level. Fine for a single tile")
    t.add_argument("--max-building-width", type=float, default=120.0, metavar="M",
                   help="DEM filter window; must exceed the widest building (default 120)")

    c = ap.add_argument_group("calibration")
    c.add_argument("--semantic", metavar="TIF",
                   help="land-cover map; enables offset calibration on ground+road pixels")
    c.add_argument("--offset", type=float, metavar="M",
                   help="add this constant to the AGL before anything else")
    c.add_argument("--no-calibrate", action="store_true",
                   help="skip the ground-pixel offset fit even if --semantic is given")

    ap.add_argument("--vertical-datum", default=DEFAULT_VERTICAL_DATUM)
    ap.add_argument("--gsd", type=float, help="metres/pixel, if the file does not say")
    a = ap.parse_args(argv)

    # ---------------------------------------------------------------- read the input
    try:
        src = read_agl(a.agl)
    except Exception as e:
        sys.exit(f"could not read {a.agl}: {e}")

    agl, valid = src["agl"], src["valid"]
    crs, transform = src["crs"], src["transform"]
    gsd = a.gsd or src["gsd"]
    h, w = agl.shape
    stem = Path(a.agl).stem
    assumptions, notes = [], []

    print(f"AGL   {w} x {h} px"
          + (f"  ({gsd} m/px" + (f", {w * gsd:.0f} x {h * gsd:.0f} m)" if gsd else ")")
             if gsd else "")
          + f"\n      georeferenced: {'yes, ' + str(crs) if crs is not None else 'no'}"
          f"\n      valid pixels:  {valid.mean() * 100:.1f}%")

    # ------------------------------------------------------------------ decide mode
    have_terrain = a.gcp or a.dem or (a.base_elevation is not None)
    if a.mode == "auto":
        mode = "absolute" if (crs is not None or have_terrain) else "relative"
    else:
        mode = a.mode
    if mode == "absolute" and not have_terrain and crs is None:
        sys.exit("--mode absolute needs a terrain source: --base-elevation, --dem or --gcp")

    # ------------------------------------------------------------- mode 1: relative
    if mode == "relative":
        print("\nmode  RELATIVE (rDSM)")
        print("      The input carries no location, so there is no way to look up a real")
        print("      ground elevation. Writing metres here would be metres you cannot")
        print("      justify, so the output is a relative surface in [0, 1] instead.")
        surface, norm = to_rdsm(agl, valid)
        assumptions.append("no spatial metadata on the input; output is relative, not metric")
        meta = write_dsm(a.out, f"{stem}_rdsm", surface, valid,
                         quantity="rDSM", units="relative",
                         crs=crs, transform=transform, gsd=gsd,
                         vertical_datum=None,
                         dtm_info={"dtm_source": "none", "method": "not applicable "
                                                                  "(relative surface)"},
                         calibration=None,
                         agl_source=src["sidecar"].get("produced_by", {}).get("checkpoint")
                                    or str(a.agl),
                         assumptions=assumptions,
                         extra={"normalisation": norm,
                                "quantity_note": "relative surface in [0,1], NOT metres. "
                                                 "For display and 3D displacement only."})
        _report(meta, mode)
        return meta

    # ------------------------------------------------------------- mode 2: absolute
    print("\nmode  ABSOLUTE DSM (metres above sea level)")

    # a manual offset, applied before everything else
    calibration = None
    if a.offset is not None:
        agl = agl + a.offset
        calibration = {"scale": 1.0, "offset": float(a.offset), "applied": True,
                       "fit": "manual", "reference": "--offset on the command line"}
        assumptions.append(f"a manual offset of {a.offset:+g} m was added to the AGL")

    # calibrate on ground pixels: they are ~0 m AGL by definition, so whatever the
    # model predicts there is its bias
    if a.semantic and not a.no_calibrate:
        sem = read_semantic(a.semantic, agl.shape)
        agl_c, rep = calibrate_to_flat_ground(agl, sem, valid=valid)
        if rep.get("applied"):
            agl = np.where(valid, agl_c, np.nan)
            print(f"      calibration: ground+road pixels read {rep['bias_before_m']:+.3f} m "
                  f"on average when they should read 0.000 m,")
            print(f"                   so {rep['offset']:+.3f} m was added "
                  f"({rep['n_reference_points']:,} reference pixels).")
            calibration = rep if calibration is None else {**calibration, "then": rep}
        else:
            notes.append("no ground/road pixels in the semantic map; nothing to calibrate on")
    elif not a.no_calibrate and a.offset is None:
        notes.append("no --semantic map, so no offset calibration was performed "
                     "(the known failure mode is a whole scene sitting a metre or two low)")

    # --------------------------------------------------------------- build the DTM
    if a.gcp:
        xs, ys, zs = read_gcps(a.gcp)
        dtm, dtm_info = gcp_plane(agl.shape, xs, ys, zs, transform=transform)
        print(f"      terrain: fitted through {len(zs)} surveyed point(s) "
              f"-- {dtm_info['method']}")
    elif a.dem:
        if crs is None or transform is None:
            sys.exit("--dem needs the AGL raster to be georeferenced, otherwise there is "
                     "no way to line the two up. Use --base-elevation instead.")
        dtm, dtm_info = load_dem_as_dtm(a.dem, agl.shape, transform, crs,
                                        max_building_width_m=a.max_building_width)
        print(f"      terrain: from {Path(a.dem).name}, buildings stripped with a "
              f"{dtm_info['window_m']:g} m opening window")
        print(f"               (that removed {dtm_info['mean_height_removed_m']:.2f} m on "
              f"average -- if that is ~0 over a city, the window is too narrow)")
        assumptions.append(
            f"DEM treated as a surface model, not bare earth: buildings removed by "
            f"morphological opening with a {dtm_info['window_m']:g} m window")
        if dtm_info.get("vertical_datum_declared_by_dem"):
            notes.append(f"the DEM declares vertical_datum="
                         f"{dtm_info['vertical_datum_declared_by_dem']!r} -- confirm it "
                         f"matches --vertical-datum")
        else:
            notes.append("the DEM file declares no vertical datum; confirm it is "
                         "orthometric (Copernicus GLO-30 is EGM2008, SRTM is EGM96)")
    else:
        base = 0.0 if a.base_elevation is None else a.base_elevation
        dtm, dtm_info = constant_plane(agl.shape, base)
        span = f"{max(h, w) * gsd:.0f} m" if gsd else "the scene"
        print(f"      terrain: flat plane at {base:g} m above sea level")
        assumptions.append(f"terrain treated as flat across the tile at {base:g} m")
        if a.base_elevation is None:
            notes.append("no --base-elevation given, so the ground was assumed to be at "
                         "0 m. The SHAPE of the surface is right; every value is offset "
                         "by the true ground elevation. Pass --base-elevation to fix it.")
        if gsd and max(h, w) * gsd > 1000:
            notes.append(f"this scene spans {span}, past the ~1 km where a flat plane "
                         f"stops being defensible. Use --dem or --gcp.")

    # ------------------------------------------------------------------- DSM = DTM + AGL
    try:
        dsm = to_dsm(agl, dtm, valid)
    except AssertionError as e:
        sys.exit(f"refusing to write: {e}")

    meta = write_dsm(a.out, f"{stem}_dsm", dsm, valid,
                     quantity="DSM", units="metres",
                     crs=crs, transform=transform, gsd=gsd,
                     vertical_datum=a.vertical_datum,
                     dtm_info=dtm_info, calibration=calibration,
                     agl_source=src["sidecar"].get("produced_by", {}).get("checkpoint")
                                or str(a.agl),
                     assumptions=assumptions, dtm=dtm,
                     extra={"quantity_note": "elevation of the top of everything, above "
                                             "the stated vertical datum. Built as "
                                             "DSM = DTM + AGL."})
    _report(meta, mode, notes)
    return meta


def _report(meta, mode, notes=()):
    v = meta["value_range"]
    unit = "m" if meta["units"] == "metres" else ""
    print(f"\nwrote {meta['files']['raster']}  +  sidecar  +  preview")
    if v["min"] is not None:
        print(f"      {meta['quantity']}  min={v['min']:.2f}{unit}  p50={v['p50']:.2f}{unit}  "
              f"max={v['max']:.2f}{unit}")
    if meta.get("terrain_range"):
        t = meta["terrain_range"]
        print(f"      terrain  {t['min']:.2f} - {t['max']:.2f} m  "
              f"(the DSM sits on top of this, never below it)")
    for n in notes:
        print(f"\n  note: {n}")
    if mode == "absolute":
        print(f"\n      datum: {meta['vertical_datum']}")
        print("      Mixing an ellipsoidal source with an orthometric one is a silent "
              "30-70 m\n      error over India. Check any file you combine with this.")


if __name__ == "__main__":
    main()
