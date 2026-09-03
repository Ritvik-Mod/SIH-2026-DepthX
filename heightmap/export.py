"""Export: the interface between this model and the rest of the team.

The single most important field is quantity="AGL".  The output is height ABOVE LOCAL
GROUND, not elevation above sea level, so the calibration stage must ADD a terrain
model, not subtract one.  Getting that backwards produces a scene that looks plausible
and is completely wrong, and no automated check will catch it.

float32 throughout.  Never 8-bit in the data path: quantisation creates flat plateaus
and tied values, which turn into degenerate critical points in the downstream
persistence computation.  The PNG is a courtesy for humans, not an interchange format.
"""
from __future__ import annotations
import json, hashlib, subprocess
from pathlib import Path
import numpy as np

SCHEMA_VERSION = "1.0"
NODATA = -9999.0


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def robust_normalise(a: np.ndarray, p_low=1.0, p_high=99.0):
    f = np.isfinite(a)
    if not f.any():
        return np.zeros_like(a), (0.0, 1.0)
    lo = float(np.percentile(a[f], p_low))
    hi = float(np.percentile(a[f], p_high))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32), (lo, hi)


def write_outputs(out_dir, stem: str, agl: np.ndarray, *, sigma=None, gsd: float | None = None,
                  crs=None, transform=None, sun=None, model_info: dict | None = None,
                  infer_meta: dict | None = None, assumptions=None, limitations=None) -> dict:
    """Writes {stem}.tif (3 bands), {stem}.json, {stem}_preview.png.  Returns the sidecar."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    agl = np.asarray(agl, np.float32)
    if sigma is None:
        sigma = np.full_like(agl, np.nan, np.float32)
    norm, (lo, hi) = robust_normalise(agl)

    bad = ~np.isfinite(agl)
    agl_w = np.where(bad, NODATA, agl).astype(np.float32)
    sig_w = np.where(np.isfinite(sigma), sigma, NODATA).astype(np.float32)
    nrm_w = np.where(bad, NODATA, norm).astype(np.float32)
    stack = np.stack([agl_w, sig_w, nrm_w])

    tif = out_dir / f"{stem}.tif"
    try:
        import rasterio
        from rasterio.transform import Affine
        prof = dict(driver="GTiff", height=agl.shape[0], width=agl.shape[1], count=3,
                    dtype="float32", nodata=NODATA, compress="deflate",
                    tiled=True, blockxsize=256, blockysize=256)
        if crs is not None:
            prof["crs"] = crs
        if transform is not None:          # omit entirely rather than writing identity,
            prof["transform"] = transform  # which GDAL warns about and may drop anyway
        with rasterio.open(tif, "w", **prof) as ds:
            ds.write(stack)
            ds.set_band_description(1, "agl_metres")
            ds.set_band_description(2, "sigma_metres")
            ds.set_band_description(3, "agl_normalised")
    except Exception as e:                                   # georeferencing optional
        np.save(out_dir / f"{stem}.npy", stack)
        tif = out_dir / f"{stem}.npy"
        print(f"  rasterio unavailable or failed ({e}); wrote {tif} instead")

    png = out_dir / f"{stem}_preview.png"
    try:
        from .utils.viz import colourise
        import imageio.v2 as imageio
        imageio.imwrite(png, colourise(np.where(bad, np.nan, agl)))
    except Exception:
        png = None

    finite = agl[np.isfinite(agl)]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "produced_by": {**(model_info or {}), "git_commit": _git_commit()},
        "quantity": "AGL",
        "quantity_note": "height above LOCAL GROUND. To obtain a DSM, ADD a terrain "
                         "model (DTM). Do not subtract.",
        "sign_convention": "larger value = taller",
        "units": "metres",
        "dtype": "float32",
        "nodata": NODATA,
        "bands": {"1": "agl_metres", "2": "sigma_metres", "3": "agl_normalised"},
        "crs": str(crs) if crs is not None else None,
        "transform": list(transform)[:6] if transform is not None else None,
        "gsd_m": float(gsd) if gsd is not None else None,
        "value_range": {
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "p1": float(np.percentile(finite, 1)) if finite.size else None,
            "p50": float(np.percentile(finite, 50)) if finite.size else None,
            "p99": float(np.percentile(finite, 99)) if finite.size else None,
        },
        "normalisation": {"method": "robust_percentile", "p_low": 1, "p_high": 99,
                          "lo": lo, "hi": hi},
        "sun": sun,
        "inference": infer_meta or {},
        "assumptions": assumptions or [],
        "known_limitations": limitations or [
            "Trained on aerial orthophotos (GAMUS, US cities); satellite and non-US "
            "scenes are out of the training distribution.",
            "Tall structures are under-represented; check the signed building bias.",
            "sigma is a model estimate combined with tile-overlap disagreement; it is "
            "calibrated on validation data, not guaranteed on unseen domains.",
        ],
        "files": {"raster": tif.name, "preview": png.name if png else None},
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta
