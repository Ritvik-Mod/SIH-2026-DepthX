"""Package inference bundles as the viewer's precomputed sample scenes.

    python scripts/package_samples.py <inference_dir> [--max-side 2048]

<inference_dir> holds one folder per scene, each the bundle the service returns
(heightmap.tif, texture.png, metadata.json). Writes, per scene:

    terrain3d/public/samples/<id>/heightmap.tif   float32, deflate + floating-point predictor
    terrain3d/public/samples/<id>/texture.jpg     same grid as the heights
    terrain3d/public/samples/<id>/thumb.jpg       gallery card
    terrain3d/public/samples/<id>/metadata.json   the service's own, updated for the above

and terrain3d/lib/samples.json, the gallery manifest.

WHY RE-ENCODE AT ALL.  The five demo scenes come back from the service as ~84 MB,
and a gallery card that takes a minute to open is not a gallery.  Three things
bring that down without changing what anyone sees:

  - scenes are capped at --max-side px.  The mesh is built from at most 1024
    segments a side, so a 3088 px raster is sampled at every third pixel anyway;
    2048 keeps the tree and building analysis, which does use every pixel,
    comfortably inside the model's trained GSD band (0.17-0.65 m/px).
  - heights are rounded to 1/128 m (7.8 mm) -- a binary fraction, so the low
    mantissa bits become exactly zero and compress to nothing.  The model's own
    error is measured in metres; 8 mm is invisible and changes no number shown.
  - the TIFF uses deflate with the floating-point predictor (TIFF TN3), which
    byte-plane-shuffles each row before differencing and is the standard way to
    make a smooth float field compressible.  geotiff.js, the viewer's reader,
    decodes it natively.

Nothing ships unchecked. The TIFF encoder below is hand-written, so verify
every scene with the viewer's own reader afterwards:

    node terrain3d/scripts/verify-samples.mjs <inference_dir>

which decodes each heightmap with geotiff.js and fails on any value that
differs from the quantised array this script saved beside the input bundle.
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "terrain3d" / "public" / "samples"
MANIFEST = ROOT / "terrain3d" / "lib" / "samples.json"
QUANT = 1 / 128
NODATA = -9999.0

# Display names and ordering. Everything else is read from the data.
SCENES = [
    {"id": "sikkim-valley-town", "name": "Sikkim valley town",
     "place": "Sikkim, India", "source": "04_nadir26_gsd0.37_valley_town.tif",
     "note": "Steep valley settlement, 26° off-nadir capture"},
    {"id": "sikkim-north", "name": "North Sikkim highlands",
     "place": "Sikkim, India", "source": "02_nadir1.8_gsd0.46_north.tif",
     "note": "High-altitude terrain above 5,200 m, near-nadir"},
    {"id": "swiss-hillside", "name": "Swiss hillside",
     "place": "Switzerland", "source": "05_geo_swiss_hillside_gsd0.33.tif",
     "note": "Houses on steep ground, swisstopo orthophoto"},
    {"id": "india-dense-urban", "name": "Dense urban block",
     "place": "India", "source": "06_nongeo_india_dense_urban.jpeg",
     "note": "Tightly packed buildings, no georeferencing"},
    {"id": "nyc-rowhouses", "name": "Row houses",
     "place": "New York, USA", "source": "07_nongeo_nyc_rowhouses.png",
     "note": "Regular terraced blocks, no georeferencing"},
]


# --------------------------------------------------------------------- TIFF

def _fp_predict(a: np.ndarray) -> np.ndarray:
    """Floating-point predictor (TIFF Technical Note 3), rows of float32.

    Each row's bytes are regrouped into planes, most significant byte first,
    then differenced left to right. geotiff.js undoes it with a running sum and
    the inverse regrouping (decodeRowFloatingPoint)."""
    h, w = a.shape
    le = np.ascontiguousarray(a, dtype="<f4").view(np.uint8).reshape(h, w, 4)
    planes = le[:, :, ::-1].transpose(0, 2, 1).reshape(h, 4 * w)  # MSB plane first
    out = np.empty_like(planes)
    out[:, 0] = planes[:, 0]
    out[:, 1:] = (planes[:, 1:].astype(np.int16) - planes[:, :-1].astype(np.int16)) & 0xFF
    return out


def write_float_tiff(path: Path, a: np.ndarray, pixel_size: float, rows_per_strip: int = 16):
    """Single-band float32 TIFF, deflate + predictor 3, with ModelPixelScale.

    Deliberately minimal. The viewer reads pixel spacing from ModelPixelScale
    and nodata from GDAL_NODATA; the full CRS stays in metadata.json, which is
    where the viewer and anything downstream actually look for it."""
    h, w = a.shape
    pred = _fp_predict(a)
    strips = [zlib.compress(pred[r:r + rows_per_strip].tobytes(), 9)
              for r in range(0, h, rows_per_strip)]

    SHORT, LONG, ASCII, DOUBLE = 3, 4, 2, 12
    nodata = b"-9999\x00"
    scale = struct.pack("<3d", pixel_size, pixel_size, 0.0)
    n = len(strips)

    tags = [  # (tag, type, count, value-or-bytes)
        (256, LONG, 1, w), (257, LONG, 1, h), (258, SHORT, 1, 32), (259, SHORT, 1, 8),
        (262, SHORT, 1, 1), (273, LONG, n, None), (277, SHORT, 1, 1),
        (278, LONG, 1, rows_per_strip), (279, LONG, n, None), (284, SHORT, 1, 1),
        (317, SHORT, 1, 3), (339, SHORT, 1, 3),
        (33550, DOUBLE, 3, scale), (42113, ASCII, len(nodata), nodata),
    ]

    # layout: header | strips | out-of-line values | IFD
    offset = 8
    strip_offsets = []
    for s in strips:
        strip_offsets.append(offset)
        offset += len(s)
    blobs = {
        273: struct.pack(f"<{n}I", *strip_offsets),
        279: struct.pack(f"<{n}I", *(len(s) for s in strips)),
        33550: scale,
        42113: nodata,
    }
    extra = bytearray()
    extra_at = {}
    for tag, typ, count, _ in tags:
        size = {SHORT: 2, LONG: 4, ASCII: 1, DOUBLE: 8}[typ] * count
        if size > 4:
            if offset + len(extra) & 1:
                extra += b"\x00"                    # values start on a word boundary
            extra_at[tag] = offset + len(extra)
            extra += blobs[tag]
    ifd_at = offset + len(extra)
    if ifd_at & 1:
        extra += b"\x00"
        ifd_at += 1

    ifd = bytearray(struct.pack("<H", len(tags)))
    for tag, typ, count, val in tags:
        ifd += struct.pack("<HHI", tag, typ, count)
        if tag in extra_at:
            ifd += struct.pack("<I", extra_at[tag])
        elif tag in blobs:                          # a single strip fits inline
            ifd += blobs[tag][:4].ljust(4, b"\x00")
        elif typ == SHORT:
            ifd += struct.pack("<HH", val, 0)
        else:
            ifd += struct.pack("<I", val)
    ifd += struct.pack("<I", 0)

    with open(path, "wb") as f:
        f.write(b"II" + struct.pack("<HI", 42, ifd_at))
        for s in strips:
            f.write(s)
        f.write(extra)
        f.write(ifd)


# -------------------------------------------------------------------- scene

def package(src: Path, scene: dict, max_side: int) -> dict:
    meta = json.loads((src / "metadata.json").read_text())
    h = np.asarray(Image.open(src / "heightmap.tif"), dtype=np.float32)
    rgb = Image.open(src / "texture.png").convert("RGB")

    bad = h <= -9000
    if bad.any():
        h[bad] = h[~bad].min()

    H, W = h.shape
    gsd = float(meta["pixel_spacing_m"])
    k = min(1.0, max_side / max(H, W))
    if k < 1.0:
        nw, nh = round(W * k), round(H * k)
        # BOX = area average: the honest way to shrink a height field.
        h = np.asarray(Image.fromarray(h).resize((nw, nh), Image.BOX), dtype=np.float32)
        rgb = rgb.resize((nw, nh), Image.LANCZOS)
        gsd = gsd * W / nw                          # same ground extent, bigger pixels
    elif rgb.size != (W, H):
        rgb = rgb.resize((W, H), Image.LANCZOS)

    h = (np.round(h / QUANT) * QUANT).astype(np.float32)
    nh, nw = h.shape

    dst = OUT / scene["id"]
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    write_float_tiff(dst / "heightmap.tif", h, gsd)
    rgb.save(dst / "texture.jpg", "JPEG", quality=90, optimize=True, progressive=True)
    thumb = rgb.copy()
    thumb.thumbnail((560, 560), Image.LANCZOS)
    thumb.save(dst / "thumb.jpg", "JPEG", quality=82, optimize=True, progressive=True)

    meta.update({
        "resolution": {"width": nw, "height": nh},
        "pixel_spacing_m": gsd,
        "min_height_m": float(h.min()),
        "max_height_m": float(h.max()),
        "mean_height_m": float(h.mean()),
        "nodata_pixels": int(bad.sum()),
        "sample": {
            "note": "Precomputed demo scene. Re-encoded for the web: see scripts/package_samples.py.",
            "source_image": scene["source"],
            "original_resolution": {"width": W, "height": H},
            "downsampled": k < 1.0,
            "height_quantum_m": QUANT,
            "texture": "JPEG q90",
        },
    })
    (dst / "metadata.json").write_text(json.dumps(meta, indent=2))

    size = sum(p.stat().st_size for p in dst.iterdir() if p.name != "thumb.jpg")
    return {
        **scene,
        "quantity": meta["quantity"],
        "georeferenced": bool(meta.get("georeferenced")),
        "crs": meta.get("crs"),
        "width": nw, "height": nh,
        "originalWidth": W, "originalHeight": H,
        "gsd": round(gsd, 4),
        "extent": [round(nw * gsd), round(nh * gsd)],
        "min": round(float(h.min()), 2),
        "max": round(float(h.max()), 2),
        "relief": round(float(h.max() - h.min()), 1),
        "scaleAssumed": bool(meta.get("scale_is_assumed")),
        "bytes": size,
        "base": f"/samples/{scene['id']}",
        "_check": h,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inference_dir", type=Path)
    ap.add_argument("--max-side", type=int, default=2048)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in SCENES:
        src = a.inference_dir / scene["id"]
        if not (src / "metadata.json").exists():
            print(f"  skip {scene['id']}: no bundle in {src}")
            continue
        e = package(src, scene, a.max_side)
        # The exact array written, for verify_samples.mjs to compare against.
        np.save(src / "packaged_heights.npy", e.pop("_check"))
        entries.append(e)
        print(f"  {e['id']:20s} {e['quantity']:4s} {e['width']}x{e['height']} "
              f"gsd {e['gsd']} relief {e['relief']} m  {e['bytes'] / 1e6:.1f} MB")

    MANIFEST.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)} ({len(entries)} scenes)")


if __name__ == "__main__":
    main()
