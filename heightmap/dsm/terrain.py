"""Where the terrain (DTM) comes from.

Three sources, in descending order of how much you should trust them:

    gcp_fit         you measured the real elevation at a few points  -> best
    dem_opening     you have a DEM file; strip the buildings off it  -> good
    constant_plane  you know roughly one elevation for the scene     -> fine for one tile

THE TRAP.  Free global elevation data (SRTM, Copernicus GLO-30) is *not* bare earth.
It is a DSM: over a city it already sits at roughly rooftop level.  Adding our AGL to
it straight would count every building twice --

    WRONG:  SRTM + AGL = rooftop + 18 m = a 36 m building

-- and the result renders perfectly and errors nowhere.  So a DEM is never used
directly here; it goes through dem_to_dtm() first, which erases the buildings.
Over bare mountains the trap does not exist (nothing stands above the ground), but
the filter is harmless there, so it is always applied.
"""
from __future__ import annotations

import numpy as np

NODATA = -9999.0


# --------------------------------------------------------------------------- helpers
def _fill_invalid(a: np.ndarray) -> np.ndarray:
    """Replace non-finite / nodata cells with the nearest valid value.

    The morphological filters below take minima over a window; a single -9999 left in
    the array would spread that sentinel across the whole neighbourhood and quietly
    drag the terrain down by kilometres.
    """
    a = np.asarray(a, np.float64)
    bad = ~np.isfinite(a) | (a == NODATA)
    if not bad.any():
        return a
    if bad.all():
        return np.zeros_like(a)
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


# ------------------------------------------------------------------- DEM -> bare earth
def dem_to_dtm(dem: np.ndarray, pixel_size_m: float, max_building_width_m: float = 120.0):
    """Approximate a bare-earth DTM from a surface DEM by morphological opening.

    Opening = minimum over a window, then maximum over the same window.  Anything
    narrower than the window and standing above its surroundings (a building) is
    erased; anything broader (a hill) survives.

    The window must be WIDER than the widest building in the scene, or the middle of a
    large flat roof survives the opening and becomes permanent terrain.  120 m is a
    reasonable default for Indian cities; a mall or an airport terminal needs more.
    Too wide is also wrong -- it flattens genuine hills.

    Returns (dtm, info).
    """
    from scipy import ndimage

    pixel_size_m = float(pixel_size_m)
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError(f"pixel_size_m must be a positive number, got {pixel_size_m!r}")

    filled = _fill_invalid(dem)
    r = max(1, int(round(max_building_width_m / pixel_size_m / 2.0)))
    size = 2 * r + 1
    ground = ndimage.grey_opening(filled, size=(size, size))       # min, then max
    # Opening leaves stair-steps at the window scale; smooth them out.  Terrain is a
    # low-frequency trend and should look like one.
    dtm = ndimage.gaussian_filter(ground, sigma=r / 2.0)

    removed = float(np.nanmean(filled - dtm))
    return dtm.astype(np.float32), {
        "method": f"morphological opening, {max_building_width_m:g} m window",
        "window_px": size,
        "window_m": round(size * pixel_size_m, 2),
        "mean_height_removed_m": round(removed, 3),
    }


def load_dem_as_dtm(dem_path, dst_shape, dst_transform, dst_crs,
                    max_building_width_m: float = 120.0):
    """Read a DEM file, strip its buildings, and resample it onto our image grid.

    The resolution mismatch matters.  A 30 m DEM against 0.33 m imagery means one DEM
    cell covers about 90x90 of ours.  Resampled with nearest-neighbour that becomes a
    visible staircase under sharp buildings, and the steps look like terrain features
    that do not exist.  Cubic, always.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(str(dem_path)) as src:
        dem = src.read(1).astype(np.float64)
        if src.nodata is not None:
            dem = np.where(dem == src.nodata, np.nan, dem)
        src_crs, src_transform = src.crs, src.transform
        dem_px = float(abs(src_transform.a))
        dem_tags = {k.lower(): v for k, v in (src.tags() or {}).items()}

    if src_crs is None or src_transform is None:
        raise ValueError(f"{dem_path} has no CRS/transform; it cannot be aligned to the image")

    # Degrees, not metres: a geographic DEM (EPSG:4326) has a pixel size like 0.00027.
    # Converting at the equator is close enough to size the opening window.
    if src_crs.is_geographic:
        dem_px_m = dem_px * 111_320.0
    else:
        dem_px_m = dem_px

    dtm_coarse, dtm_info = dem_to_dtm(dem, dem_px_m, max_building_width_m)

    dtm_fine = np.empty(dst_shape, np.float32)
    reproject(
        source=dtm_coarse, destination=dtm_fine,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.cubic,                # NOT nearest
    )

    info = {
        "dtm_source": "dem_file",
        "dem_path": str(dem_path),
        "dem_pixel_size_m": round(dem_px_m, 3),
        "resampling": "cubic",
        "vertical_datum_declared_by_dem": dem_tags.get("vertical_datum"),
        **dtm_info,
    }
    return dtm_fine, info


# ------------------------------------------------------------------- constant plane
def constant_plane(shape, base_elevation_m: float):
    """The whole terrain is one number.

    Defensible, not a shortcut.  A 1024x1024 tile at 0.33 m/px is 338 x 338 metres.
    Across 338 m of a city the ground rises or falls by maybe one or two metres --
    smaller than the height model's own error bar (1.585 m).  So a flat plane costs
    less accuracy than it looks like it should.

    It stops being defensible past roughly 1 km, or anywhere with real relief:
    hill stations, coastal cliffs, river valleys.
    """
    base = float(base_elevation_m)
    dtm = np.full(shape, base, np.float32)
    return dtm, {
        "dtm_source": "constant_plane",
        "method": "constant plane",
        "base_elevation_m": base,
    }


# ---------------------------------------------------------------------- GCP plane fit
def gcp_plane(shape, xs, ys, elevations, transform=None):
    """Fit elevation = a*x + b*y + c through known points and evaluate it everywhere.

    With 3 or more points you recover tilt as well as offset, which is why this is the
    most accurate option available without a DEM.  With fewer than 3 (or with points
    that are collinear, which makes the fit ill-posed) it degrades to a flat plane at
    their mean elevation rather than extrapolating a wild slope off two samples.

    xs / ys are in the same space as `transform` maps to when one is given (world
    coordinates), otherwise pixel column / row.
    """
    xs = np.asarray(xs, np.float64).ravel()
    ys = np.asarray(ys, np.float64).ravel()
    zs = np.asarray(elevations, np.float64).ravel()
    if not (xs.size == ys.size == zs.size) or xs.size == 0:
        raise ValueError("gcp x, y and elevation must be non-empty and the same length")

    h, w = shape
    cols, rows = np.meshgrid(np.arange(w, dtype=np.float64),
                             np.arange(h, dtype=np.float64))
    if transform is not None:
        X = transform.c + transform.a * (cols + 0.5) + transform.b * (rows + 0.5)
        Y = transform.f + transform.d * (cols + 0.5) + transform.e * (rows + 0.5)
    else:
        X, Y = cols, rows

    if xs.size >= 3:
        A = np.column_stack([xs, ys, np.ones_like(xs)])
        if np.linalg.matrix_rank(A) == 3:
            coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
            dtm = coef[0] * X + coef[1] * Y + coef[2]
            resid = zs - (coef[0] * xs + coef[1] * ys + coef[2])
            return dtm.astype(np.float32), {
                "dtm_source": "gcp_fit",
                "method": "least-squares tilted plane through ground control points",
                "n_points": int(xs.size),
                "coefficients": [float(c) for c in coef],
                "rms_residual_m": round(float(np.sqrt(np.mean(resid ** 2))), 3),
            }

    base = float(np.mean(zs))
    dtm = np.full(shape, base, np.float32)
    return dtm, {
        "dtm_source": "gcp_fit",
        "method": "flat plane at mean of control points (too few / collinear for a tilt)",
        "n_points": int(xs.size),
        "base_elevation_m": base,
    }
