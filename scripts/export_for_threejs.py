#!/usr/bin/env python
"""Package one predicted AGL field + its RGB tile for a Three.js terrain viewer.

Writes into export/:
    heightmap.tif          single-band float32 GeoTIFF, REAL METRES, no scaling
    texture.png            the RGB, matched pixel-for-pixel to the heightmap
    heightmap_preview.png  8-bit grey, normalised, for eyeballing only
    metadata.json          ranges + spacing, for display; NOT for denormalising

The quantity in heightmap.tif is AGL -- height above LOCAL GROUND, not elevation above
sea level.  A DSM is DTM + AGL.  Adding terrain is correct; subtracting it produces a
scene that looks plausible and is completely wrong, and nothing downstream will catch
it.  See SPEC.md.

Nodata is -9999.0, matching heightmap/export.py.  Mask it before any statistics: it is
a sentinel, not a height, and averaging it in shifts a mean by kilometres.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image

NODATA = -9999.0          # keep in step with heightmap/export.py
DEFAULT_GSD = 0.33        # metres/pixel, GAMUS paper sec.1


def load_agl(path: str) -> np.ndarray:
    a = np.load(path)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D height field, got shape {a.shape} from {path}")
    return a.astype(np.float32)


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def match_to(rgb: np.ndarray, shape_hw: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Bring the texture to exactly the heightmap's pixel dimensions.

    Aspect ratio is handled by CENTRE-CROPPING first, then resizing.  Resizing a
    different aspect ratio directly would stretch the photo relative to the geometry,
    so a roof would sit beside its own footprint in the render -- a misalignment that
    looks like a model error and is not one.
    """
    H, W = shape_hw
    h, w = rgb.shape[:2]
    if (h, w) == (H, W):
        return rgb, "already matched"

    note = []
    if abs((w / h) - (W / H)) > 1e-6:
        target = W / H
        if (w / h) > target:                       # too wide -> trim left/right
            new_w = int(round(h * target)); x0 = (w - new_w) // 2
            rgb = rgb[:, x0:x0 + new_w]
        else:                                      # too tall -> trim top/bottom
            new_h = int(round(w / target)); y0 = (h - new_h) // 2
            rgb = rgb[y0:y0 + new_h, :]
        note.append(f"centre-cropped {w}x{h} -> {rgb.shape[1]}x{rgb.shape[0]}")

    if rgb.shape[:2] != (H, W):
        resample = Image.LANCZOS if rgb.shape[0] > H else Image.BICUBIC
        rgb = np.asarray(Image.fromarray(rgb).resize((W, H), resample), dtype=np.uint8)
        note.append(f"resized to {W}x{H}")
    return rgb, "; ".join(note)


def write_heightmap(path: Path, agl: np.ndarray, *, crs=None, transform=None) -> str:
    """Single-band float32 GeoTIFF holding real metres.

    crs/transform are already plumbed through and simply omitted when None, so a real
    georeferenced input later is a two-argument call rather than a rewrite.  They are
    omitted rather than set to identity: GDAL warns on an identity transform and may
    drop it, and a bogus transform is worse than none -- a consumer cannot tell it apart
    from a real one.
    """
    import rasterio
    prof = dict(driver="GTiff", height=agl.shape[0], width=agl.shape[1], count=1,
                dtype="float32", nodata=NODATA, compress="deflate",
                tiled=True, blockxsize=256, blockysize=256)
    if crs is not None:
        prof["crs"] = crs
    if transform is not None:
        prof["transform"] = transform
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(agl.astype(np.float32), 1)
        ds.set_band_description(1, "agl_metres")
        ds.update_tags(quantity="AGL",
                       quantity_note="height above LOCAL GROUND. DSM = DTM + this. "
                                     "ADD the terrain model, do not subtract.")
    return prof["dtype"]


def write_preview(path: Path, agl: np.ndarray, valid: np.ndarray,
                  p_low=1.0, p_high=99.0) -> None:
    """8-bit grey, percentile-stretched.  Viewing aid only -- never a data source.

    Stretched on percentiles rather than min/max because one 390 m outlier (GAMUS has
    such a tile) would push every building to the bottom of the ramp and the preview
    would read as an empty field.
    """
    out = np.zeros(agl.shape, np.uint8)
    if valid.any():
        v = agl[valid]
        lo, hi = np.percentile(v, [p_low, p_high])
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        out[valid] = (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)


def export(agl_path: str, rgb_path: str, out_dir: str = "export",
           gsd: float = DEFAULT_GSD) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    agl = load_agl(agl_path)
    rgb = load_rgb(rgb_path)
    H, W = agl.shape

    # A prediction should have no nodata, but a field read back from a GeoTIFF can.
    # Treat non-finite and the sentinel identically, and keep statistics off both.
    valid = np.isfinite(agl) & (agl != NODATA)
    agl_w = np.where(valid, agl, NODATA).astype(np.float32)

    rgb_m, match_note = match_to(rgb, (H, W))
    Image.fromarray(rgb_m).save(out / "texture.png")
    dtype = write_heightmap(out / "heightmap.tif", agl_w)
    write_preview(out / "heightmap_preview.png", agl, valid)

    v = agl[valid]
    meta = {
        "quantity": "AGL",
        "quantity_note": "height above LOCAL GROUND, in metres. DSM = DTM + heightmap. "
                         "ADD the terrain model, do not subtract.",
        "units": "metres",
        "min_height_m": float(v.min()) if v.size else None,
        "max_height_m": float(v.max()) if v.size else None,
        "mean_height_m": float(v.mean()) if v.size else None,
        "resolution": {"width": int(W), "height": int(H)},
        "pixel_spacing_m": float(gsd),
        "ground_extent_m": {"width": round(W * gsd, 2), "height": round(H * gsd, 2)},
        "geotiff_dtype": dtype,
        "nodata": NODATA,
        "nodata_pixels": int((~valid).sum()),
        "georeferenced": False,
        "note": "heightmap.tif already holds real metres. These values are for display "
                "and sanity checks only -- do NOT denormalise against them. "
                "heightmap_preview.png IS normalised and is not a data source.",
        "source": {"agl": str(agl_path), "rgb": str(rgb_path)},
        "texture_match": match_note or "already matched",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Package AGL + RGB for a Three.js terrain.")
    ap.add_argument("agl_npy", help="predicted AGL, .npy, (H,W) float metres")
    ap.add_argument("rgb_image", help="matching RGB tile (png/jpg/tif)")
    ap.add_argument("--out", default="export")
    ap.add_argument("--gsd", type=float, default=DEFAULT_GSD, help="metres per pixel")
    a = ap.parse_args(argv)

    m = export(a.agl_npy, a.rgb_image, a.out, a.gsd)
    o = Path(a.out)
    print(f"\nexported to {o}/")
    for f in ("heightmap.tif", "texture.png", "heightmap_preview.png", "metadata.json"):
        p = o / f
        print(f"  {f:24s} {p.stat().st_size/1024:9.1f} KB")
    print(f"\n  quantity        AGL (height above LOCAL ground) -- DSM = DTM + heightmap")
    print(f"  resolution      {m['resolution']['width']} x {m['resolution']['height']} px"
          f"  @ {m['pixel_spacing_m']} m/px"
          f"  = {m['ground_extent_m']['width']} x {m['ground_extent_m']['height']} m")
    print(f"  height range    {m['min_height_m']:.2f} .. {m['max_height_m']:.2f} m"
          f"   mean {m['mean_height_m']:.2f} m")
    print(f"  geotiff dtype   {m['geotiff_dtype']}   nodata {m['nodata']}"
          f"   ({m['nodata_pixels']} px)")
    print(f"  texture         {m['texture_match']}")
    print(f"  georeferenced   {m['georeferenced']} (no CRS/transform written)")
    print("\n  heightmap.tif holds REAL METRES. Do not rescale it. The preview PNG is\n"
          "  normalised for viewing only and must not be used as geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
