"""AGL -> DSM.  The stage that turns "how tall is this thing" into "how high is its top".

    DSM  =  DTM  +  AGL
            ^       ^
         terrain   what the height model gives us

ADD the terrain.  Never subtract it.  If you compute DTM - AGL the scene still looks
like a plausible landscape, renders fine, and is completely wrong -- no automated check
catches it, which is why assert_dsm_above_dtm() below is called on every write.

Two modes, decided by what the input carries (PS requirements #5 and #6):

    plain PNG / JPG, no spatial metadata  ->  rDSM, a relative surface in [0, 1]
    GeoTIFF with a CRS and a transform    ->  absolute DSM in metres above a datum

The second needs a terrain elevation; the first cannot have one, because without a
location there is no way to look one up.  Writing metres you cannot justify is worse
than writing an honest relative surface, so mode 1 does not pretend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NODATA = -9999.0
SCHEMA_VERSION = "1.0"

#: What "zero elevation" means.  Copernicus GLO-30 is EGM2008; SRTM is EGM96; raw GPS
#: is ellipsoidal WGS84.  Mixing an ellipsoidal reference with an orthometric DEM is a
#: silent 30-70 m error over India, with no warning and a perfectly normal render.
DEFAULT_VERTICAL_DATUM = "EGM2008 (orthometric, approx. mean sea level)"


# ------------------------------------------------------------------------------ input
def read_agl(path):
    """Read an AGL raster produced by this project.

    Accepts the 3-band contract raster (band 1 = agl_metres), a plain 1-band heightmap,
    or the .npy fallback that export.py writes when rasterio is unavailable.

    Never read heightmap_preview.png.  It is 8-bit and percentile-stretched for human
    eyes: on a nearly flat scene it turns 1 m of noise into what looks like a mountain
    range.  Heights read from it are fiction.

    Returns a dict with agl, valid, crs, transform, gsd, sidecar.
    """
    path = Path(path)
    if path.suffix.lower() == ".png":
        raise ValueError(
            f"{path.name} is a preview image: 8-bit and percentile-stretched, so the "
            "values in it are not metres. Read the .tif instead."
        )

    crs = transform = None
    gsd = None
    if path.suffix.lower() == ".npy":
        stack = np.load(path)
        agl = np.asarray(stack[0] if stack.ndim == 3 else stack, np.float64)
        file_nodata = NODATA
    else:
        import rasterio
        with rasterio.open(str(path)) as ds:
            agl = ds.read(1).astype(np.float64)
            crs, transform = ds.crs, ds.transform
            file_nodata = ds.nodata if ds.nodata is not None else NODATA
            if transform is not None and not transform.is_identity and abs(transform.a) > 0:
                gsd = float(abs(transform.a))
            else:
                crs = transform = None          # identity transform is not georeferencing

    valid = np.isfinite(agl) & (agl != file_nodata) & (agl != NODATA)
    agl = np.where(valid, agl, np.nan)

    # The sidecar is authoritative for GSD when the raster is not georeferenced.
    sidecar = {}
    side = path.with_suffix(".json")
    if side.exists():
        try:
            sidecar = json.loads(side.read_text())
        except Exception:
            sidecar = {}
        if gsd is None and sidecar.get("gsd_m"):
            gsd = float(sidecar["gsd_m"])
        q = sidecar.get("quantity")
        if q and q != "AGL":
            raise ValueError(
                f"{side.name} says quantity={q!r}, not 'AGL'. This stage converts AGL to "
                "a DSM; feeding it something that is already a DSM would add terrain twice."
            )

    return {"agl": agl, "valid": valid, "crs": crs, "transform": transform,
            "gsd": gsd, "sidecar": sidecar, "path": path}


# ------------------------------------------------------------------ mode 1: relative
def to_rdsm(agl, valid, p_low=1.0, p_high=99.0):
    """Relative DSM in [0, 1] -- PS requirement #5, for input with no spatial metadata.

    Robust percentiles, not min/max.  One 390 m outlier (GAMUS has such a tile) would
    otherwise push every building to the bottom of the range and the whole scene would
    render flat.
    """
    agl = np.asarray(agl, np.float64)
    valid = np.asarray(valid, bool)
    v = agl[valid & np.isfinite(agl)]
    if v.size == 0:
        return np.zeros(agl.shape, np.float32), {"lo": 0.0, "hi": 1.0}
    lo, hi = (float(x) for x in np.percentile(v, [p_low, p_high]))
    r = np.clip((agl - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out = np.where(valid & np.isfinite(agl), r, 0.0).astype(np.float32)
    return out, {"method": "robust_percentile", "p_low": p_low, "p_high": p_high,
                 "lo": lo, "hi": hi}


# ------------------------------------------------------------------ mode 2: absolute
def assert_dsm_above_dtm(dsm, dtm, valid, agl=None, neg_tol=0.5):
    """The one check that catches a subtracted terrain.

    If terrain were subtracted instead of added, the DSM would sit *below* the DTM by a
    full building height across every building.  Nothing else catches that: the numbers
    stay in a plausible range and the render looks fine.

    Two checks, because they fail differently:

    1. If `agl` is supplied, (dsm - dtm) must equal it.  A sign error shows up here as a
       discrepancy of 2x the building height, regardless of the terrain's own values.
    2. Otherwise, the DSM must not dip more than `neg_tol` below the DTM.

    `neg_tol` is 0.5 m rather than 0 on purpose.  AGL after offset calibration is
    legitimately a few centimetres negative over ground -- that is the calibration
    working, not a bug -- while a subtracted terrain is wrong by metres.  A tolerance
    tight enough to reject the former would fire constantly and be switched off, which
    is the failure mode this check exists to prevent.
    """
    v = np.asarray(valid, bool) & np.isfinite(dsm) & np.isfinite(dtm)
    if not v.any():
        return
    dsm = np.asarray(dsm, np.float64)
    dtm = np.asarray(dtm, np.float64)

    if agl is not None:
        agl = np.asarray(agl, np.float64)
        v2 = v & np.isfinite(agl)
        if v2.any():
            err = float(np.max(np.abs((dsm[v2] - dtm[v2]) - agl[v2])))
            if err > max(neg_tol, 1e-3):
                raise AssertionError(
                    f"DSM - DTM does not equal AGL (worst discrepancy {err:.3f} m). "
                    "DSM = DTM + AGL -- terrain is ADDED, never subtracted."
                )

    diff = dsm[v] - dtm[v]
    below = diff < -neg_tol
    if below.any():
        raise AssertionError(
            f"DSM sits more than {neg_tol} m below the DTM at {below.sum()} of {v.sum()} "
            f"valid pixels (worst {float(diff.min()):.3f} m). DSM = DTM + AGL -- terrain "
            "is ADDED, never subtracted."
        )


def to_dsm(agl, dtm, valid):
    """DSM = DTM + AGL.  Checked, not assumed."""
    agl = np.asarray(agl, np.float64)
    dtm = np.asarray(dtm, np.float64)
    dsm = np.where(valid, dtm + agl, np.nan)
    assert_dsm_above_dtm(dsm, dtm, valid, agl=agl)
    return dsm.astype(np.float32)


# ----------------------------------------------------------------------------- output
def write_dsm(out_dir, stem, surface, valid, *, quantity, units, crs=None, transform=None,
              gsd=None, vertical_datum=None, dtm_info=None, calibration=None,
              agl_source=None, assumptions=None, extra=None, dtm=None):
    """Write {stem}.tif (float32, 1 band, nodata -9999, DEFLATE, tiled 256) + {stem}.json.

    The sidecar's `assumptions` array is not decoration.  Every shortcut taken belongs
    in it: a stated assumption is engineering, an unstated one is a bug waiting for
    someone else to find.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surface = np.asarray(surface, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(surface)
    band = np.where(valid, surface, NODATA).astype(np.float32)

    tif = out_dir / f"{stem}.tif"
    wrote_raster = True
    try:
        import rasterio
        prof = dict(driver="GTiff", height=band.shape[0], width=band.shape[1], count=1,
                    dtype="float32", nodata=NODATA, compress="deflate",
                    tiled=True, blockxsize=256, blockysize=256)
        if crs is not None:
            prof["crs"] = crs
        if transform is not None:
            prof["transform"] = transform
        with rasterio.open(tif, "w", **prof) as ds:
            ds.write(band, 1)
            ds.set_band_description(1, f"{quantity.lower()}_{units}")
            tags = {"quantity": quantity, "units": units}
            if vertical_datum:
                tags["vertical_datum"] = vertical_datum
            if dtm_info:
                tags["dtm_source"] = str(dtm_info.get("dtm_source", ""))
                tags["dtm_method"] = str(dtm_info.get("method", ""))
                if "base_elevation_m" in dtm_info:
                    tags["base_elevation_m"] = str(dtm_info["base_elevation_m"])
            ds.update_tags(**tags)
    except Exception as e:
        np.save(out_dir / f"{stem}.npy", band)
        tif = out_dir / f"{stem}.npy"
        wrote_raster = False
        print(f"  rasterio unavailable or failed ({e}); wrote {tif.name} instead")

    png = out_dir / f"{stem}_preview.png"
    try:
        import imageio.v2 as imageio
        from ..utils.viz import colourise
        imageio.imwrite(png, colourise(np.where(valid, surface, np.nan)))
    except Exception:
        png = None

    f = surface[valid]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "quantity": quantity,
        "units": units,
        "dtype": "float32",
        "nodata": NODATA,
        "sign_convention": "larger value = higher",
        "vertical_datum": vertical_datum,
        "georeferenced": bool(crs is not None and transform is not None),
        "crs": str(crs) if crs is not None else None,
        "transform": list(transform)[:6] if transform is not None else None,
        "pixel_spacing_m": float(gsd) if gsd else None,
        "dtm_source": (dtm_info or {}).get("dtm_source"),
        "dtm_method": (dtm_info or {}).get("method"),
        "dtm_details": dtm_info or None,
        "calibration": calibration,
        "agl_source": agl_source,
        "value_range": {
            "min": float(f.min()) if f.size else None,
            "max": float(f.max()) if f.size else None,
            "p1": float(np.percentile(f, 1)) if f.size else None,
            "p50": float(np.percentile(f, 50)) if f.size else None,
            "p99": float(np.percentile(f, 99)) if f.size else None,
        },
        "valid_pixel_fraction": round(float(valid.mean()), 4),
        "assumptions": assumptions or [],
        "files": {"raster": tif.name, "preview": png.name if png else None},
    }
    if dtm is not None:
        d = np.asarray(dtm, np.float64)[valid]
        if d.size:
            meta["terrain_range"] = {"min": float(d.min()), "max": float(d.max()),
                                     "mean": float(d.mean())}
    if extra:
        meta.update(extra)

    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    meta["_raster_path"] = str(tif)
    meta["_wrote_geotiff"] = wrote_raster
    return meta
