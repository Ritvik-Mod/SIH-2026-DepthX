#!/usr/bin/env python
"""ONE image in -> everything Three.js needs out.  The connector.

    python scripts/run_pipeline.py --ckpt a7_vitl_v0.1.0.pt --image scene.tif --out run1/

Before this existed the path was three tools that did not quite join up:
heightmap.predict wrote a 3-band .tif, while scripts/export_for_threejs.py expected a
.npy, so "run the model and hand the result to the viewer" was a manual step nobody
had written down.  This is that step.

    input image ──> heightmap.infer.predict_scene ──> AGL (metres)
                          │
                          ├─ optional DEM ──> heightmap.dsm ──> DSM        (Divyanshu)
                          │
                          └─> bundle/  heightmap.tif   single-band float32, real metres
                                       texture.png     RGB, pixel-matched to the raster
                                       metadata.json   spacing, ranges, provenance
                                       heightmap_preview.png   humans only
                              plus the full SPEC.md sidecar bundle in raw/

manifest.json at the top level is the machine-readable summary: an HTTP service, a
batch job or a person can read it without parsing stdout.

WHICH FIELD GOES INTO heightmap.tif is a real decision, not a detail.  The viewer draws
the raster as displacement from zero, so AGL (ground at 0, buildings above it) renders
correctly out of the box.  A DSM puts the entire scene at its true elevation -- 215 m
over Delhi -- and a viewer that is not expecting that renders a flat slab 215 m in the
air.  So --render-quantity defaults to agl, dsm is available, and metadata.json always
states which one is in the file.
"""
import os, sys, json, argparse, shutil, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

from heightmap.predict import load_model, read_image
from heightmap.infer import predict_scene
from heightmap.export import write_outputs
from heightmap.utils.misc import pick_device
from heightmap import dsm as dsm_stage          # Divyanshu's AGL -> DSM stage
from export_for_threejs import match_to, write_heightmap, write_preview, NODATA


def build_bundle(out_dir: Path, field: np.ndarray, rgb: np.ndarray, gsd: float,
                 quantity: str, crs=None, transform=None, extra: dict | None = None) -> dict:
    """The four files the viewer reads.  Layout identical to export_for_threejs.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = field.shape
    valid = np.isfinite(field) & (field != NODATA)
    fw = np.where(valid, field, NODATA).astype(np.float32)

    rgb_m, match_note = match_to(rgb, (H, W))
    Image.fromarray(rgb_m).save(out_dir / "texture.png")
    write_heightmap(out_dir / "heightmap.tif", fw, crs=crs, transform=transform)
    write_preview(out_dir / "heightmap_preview.png", field, valid)

    v = field[valid]
    meta = {
        "quantity": quantity,
        "quantity_note": {
            "AGL": "height above LOCAL GROUND, in metres. Ground is 0. DSM = DTM + this.",
            "DSM": "elevation of the visible surface above the DEM's vertical datum. "
                   "The whole scene sits at its true elevation; subtract "
                   "min_height_m before rendering if your viewer expects 0 = ground.",
            "rDSM": "RELATIVE height, 0-1, NOT metres. The input carried no "
                    "georeferencing, so absolute scale is unavailable.",
        }[quantity],
        "units": "metres" if quantity != "rDSM" else "relative (0-1)",
        "min_height_m": float(v.min()) if v.size else None,
        "max_height_m": float(v.max()) if v.size else None,
        "mean_height_m": float(v.mean()) if v.size else None,
        "resolution": {"width": int(W), "height": int(H)},
        "pixel_spacing_m": float(gsd),
        "ground_extent_m": {"width": round(W * gsd, 2), "height": round(H * gsd, 2)},
        "geotiff_dtype": "float32",
        "nodata": NODATA,
        "nodata_pixels": int((~valid).sum()),
        "georeferenced": crs is not None,
        "crs": str(crs) if crs is not None else None,
        "note": "heightmap.tif holds real values -- do NOT denormalise against these "
                "numbers. heightmap_preview.png IS normalised and is not a data source.",
        "texture_match": match_note or "already matched",
    }
    meta.update(extra or {})
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def run(a, preloaded=None) -> dict:
    """preloaded=(model, cfg, device) skips the 4.6 s checkpoint load.

    A long-running service loads the model once at startup and passes it in on every
    request; the CLI passes nothing and loads per call.  Same code path either way, so
    the served result cannot drift from the command-line result.
    """
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if preloaded is not None:
        model, cfg, device = preloaded
    else:
        device = pick_device() if a.device == "auto" else a.device
        model, cfg = load_model(a.ckpt, device)

    rgb, gsd_file, crs, transform = read_image(a.image)
    gsd = a.gsd or gsd_file
    relative = gsd is None
    if relative:
        gsd = a.assume_gsd or float(cfg.data.gsd_base)
    stem = Path(a.image).stem
    print(f"[1/4] {rgb.shape[1]}x{rgb.shape[0]} px  gsd={gsd} m/px  "
          f"georeferenced={crs is not None}  device={device}"
          + ("  RELATIVE MODE" if relative else ""))

    # ---- 2. inference ----------------------------------------------------------
    t0 = time.time()
    res = predict_scene(model, rgb, gsd, tile=int(cfg.infer.tile),
                        overlap=float(cfg.infer.overlap), batch=a.batch,
                        tta=a.tta, level=not a.no_level, device=device,
                        progress=a.progress)
    t_infer = time.time() - t0
    print(f"[2/4] inference {t_infer:.1f} s over {res.meta['n_tiles']} tiles  "
          f"AGL p50={np.median(res.agl):.2f} p99={np.percentile(res.agl, 99):.2f} m")

    # ---- 3. terrain: hand off to the DSM stage ---------------------------------
    # heightmap/dsm/ is Divyanshu's stage and owns every terrain decision -- which is
    # why this reads as three calls and no arithmetic.  The double-counting trap (a
    # DEM is a SURFACE model, so its posts already sit on the rooftops) is handled
    # inside load_dem_as_dtm, and to_dsm() asserts DSM - DTM == AGL on every call,
    # which is the check that catches a subtracted terrain.
    dsm, dtm, dtm_info = None, None, None
    valid = np.isfinite(res.agl)
    dem_path = a.dem
    if a.auto_dem and not dem_path and crs is not None:
        dem_path = str(out / "dem.tif")
        import subprocess
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "fetch_dem.py"),
               "--image", a.image, "--out", dem_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[3/4] --auto-dem failed, continuing with AGL only:\n"
                  f"      {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '?'}")
            dem_path = None

    if relative:
        print("[3/4] no georeferencing: the metric field is conditional on the assumed "
              "GSD;\n      raw/ also carries an rDSM, which is the unconditional product")
    elif dem_path and crs is not None:
        dtm, dtm_info = dsm_stage.load_dem_as_dtm(dem_path, res.agl.shape, transform, crs,
                                                  max_building_width_m=a.max_structure)
        dsm = dsm_stage.to_dsm(res.agl, dtm, valid)
        print(f"[3/4] DSM from {Path(dem_path).name}: buildings stripped with a "
              f"{dtm_info['window_m']:g} m opening "
              f"(removed {dtm_info['mean_height_removed_m']:+.2f} m on average)")
        print(f"      surface {np.nanpercentile(dsm, 1):.1f}..{np.nanpercentile(dsm, 99):.1f} m")
    elif a.terrain_mode == "flat":
        dtm, dtm_info = dsm_stage.constant_plane(res.agl.shape, a.base_elevation)
        dsm = dsm_stage.to_dsm(res.agl, dtm, valid)
        print(f"[3/4] DSM on a flat plane at {a.base_elevation:g} m")
    elif dem_path:
        print("[3/4] a DEM needs a georeferenced input; skipping DSM")
    else:
        print("[3/4] no DEM: AGL only (pass --dem, --auto-dem, or --terrain-mode flat)")

    # ---- 4. write both the SPEC bundle and the viewer bundle -------------------
    raw_dir = out / "raw"
    sigma = res.sigma if res.sigma is not None else res.overlap_disagreement
    model_info = {"model": "HeightNet", "checkpoint": str(a.ckpt),
                  "backbone": str(cfg.model.checkpoint), "head": str(cfg.model.head)}
    gsd_note = (f"GSD ASSUMED to be {gsd} m/px; the input carried none, so these metres "
                f"are conditional on that guess") if relative else f"GSD {gsd} m/px"

    # the height stage writes AGL through its own contract (SPEC.md, 3 bands)
    write_outputs(raw_dir, stem, res.agl, sigma=sigma, gsd=gsd,
                  crs=None if relative else crs, transform=None if relative else transform,
                  model_info=model_info, infer_meta=res.meta, assumptions=[gsd_note])

    # the DSM stage writes its own outputs through SPEC_DSM.md (1 band) -- two
    # contracts, each owned by the stage that produces it, rather than one file
    # whose meaning changes depending on the flags it was made with
    if relative:
        rdsm, rinfo = dsm_stage.to_rdsm(res.agl, valid)
        dsm_stage.write_dsm(raw_dir, f"{stem}_rdsm", rdsm, valid,
                            quantity="rDSM", units="relative (0-1)", gsd=gsd,
                            agl_source=str(a.image),
                            assumptions=[gsd_note,
                                         "absolute elevation is NOT recoverable from this "
                                         "output; rank and correlation are valid, "
                                         "differences in metres are not"],
                            extra={"normalisation": rinfo})
    elif dsm is not None:
        dsm_stage.write_dsm(raw_dir, f"{stem}_dsm", dsm, valid,
                            quantity="DSM", units="metres", crs=crs, transform=transform,
                            gsd=gsd, vertical_datum=dsm_stage.DEFAULT_VERTICAL_DATUM,
                            dtm_info=dtm_info, agl_source=str(a.image), dtm=dtm,
                            assumptions=[gsd_note,
                                         "vertical datum inherited from the terrain "
                                         "source and NOT converted"])

    want = a.render_quantity
    if want == "dsm" and dsm is None:
        print("      --render-quantity dsm requested but no DSM was produced; using AGL")
        want = "agl"
    if relative:
        # The RENDER gets metric AGL under the declared GSD assumption; the DATA
        # deliverable in raw/ stays rDSM.  A 0-1 field is the honest data product but
        # an unusable render: 1 metre of relief across a 169 m scene draws as a flat
        # plate.  Same split the TDA work uses -- render path and accuracy path are
        # separate deliverables -- and metadata.json flags the scale as assumed so the
        # metres in the raster can never be mistaken for measured ones.
        field, quantity = res.agl, "AGL"
    elif want == "dsm":
        field, quantity = dsm, "DSM"
    else:
        field, quantity = res.agl, "AGL"

    meta = build_bundle(out / "bundle", field, rgb, gsd, quantity,
                        crs=crs if not relative else None,
                        transform=transform if not relative else None,
                        extra={"source_image": Path(a.image).name,
                               "terrain": dtm_info,
                               "produced_by": model_info,
                               "inference": res.meta,
                               "scale_is_assumed": bool(relative),
                               "assumed_gsd_m": gsd if relative else None,
                               "scale_note": (
                                   f"The input carried no georeferencing. These metres "
                                   f"are conditional on an ASSUMED {gsd} m/px and are "
                                   f"valid for rendering and for relative comparison, "
                                   f"not as measured elevations. The unconditional "
                                   f"product is raw/{stem}.tif (quantity rDSM)."
                                   ) if relative else None})
    manifest = {
        "ok": True,
        "stem": stem,
        "quantity": quantity,
        "georeferenced": crs is not None and not relative,
        "relative_mode": relative,
        "gsd_m": gsd,
        "gsd_source": "file" if gsd_file else ("cli" if a.gsd else "assumed"),
        "resolution": meta["resolution"],
        "height_range_m": [meta["min_height_m"], meta["max_height_m"]],
        "has_dsm": dsm is not None,
        "timing_s": {"inference": round(t_infer, 2), "total": round(time.time() - t_start, 2)},
        "bundle": {n: f"bundle/{n}" for n in
                   ("heightmap.tif", "texture.png", "metadata.json", "heightmap_preview.png")},
        "raw_dir": "raw",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[4/4] wrote {out}/bundle/ + {out}/raw/  in {manifest['timing_s']['total']:.1f} s total")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="image -> AGL/DSM -> Three.js bundle")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/pipeline")
    ap.add_argument("--gsd", type=float, default=None)
    ap.add_argument("--assume-gsd", type=float, default=None)
    ap.add_argument("--dem", default=None)
    ap.add_argument("--auto-dem", action="store_true",
                    help="fetch GLO-30 for the footprint (needs internet + a CRS)")
    ap.add_argument("--terrain-mode", default="dem", choices=["dem", "flat"],
                    help="dem: strip structures out of --dem and use it as ground. "
                         "flat: a constant plane at --base-elevation, no DEM needed.")
    ap.add_argument("--max-structure", type=float, default=120.0,
                    help="metres; widest building the DEM opening should remove")
    ap.add_argument("--base-elevation", type=float, default=0.0)
    ap.add_argument("--render-quantity", default="agl", choices=["agl", "dsm"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-level", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args(argv)
    m = run(a)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
