"""CLI: one image in, a height raster out.

    python -m heightmap.predict --ckpt outputs/a5/best.pt --image scene.tif --out outputs/pred

Accepts GeoTIFF (CRS/transform/GSD carried through) or PNG/JPG (GSD must be supplied).
Writes the three files described in SPEC.md.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

from . import config
from .models.net import HeightNet
from .infer import predict_scene
from .export import write_outputs
from .utils.misc import pick_device
from omegaconf import OmegaConf


def read_image(path):
    """-> (rgb uint8 HxWx3, gsd or None, crs or None, transform or None)"""
    p = str(path)
    if p.lower().endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(p) as ds:
            arr = ds.read()
            if arr.shape[0] < 3:
                arr = np.repeat(arr[:1], 3, 0)
            rgb = arr[:3].transpose(1, 2, 0)
            if rgb.dtype != np.uint8:                    # 11/16-bit -> 8-bit for the encoder
                f = rgb.astype(np.float32)
                lo, hi = np.percentile(f, 1), np.percentile(f, 99)
                rgb = (np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
            t = ds.transform
            gsd = float(abs(t.a)) if t is not None and abs(t.a) > 0 else None
            crs = ds.crs
            has_geo = crs is not None and t is not None and not t.is_identity
            return rgb, (gsd if has_geo else None), (crs if has_geo else None), (t if has_geo else None)
    from PIL import Image
    return np.array(Image.open(p).convert("RGB")), None, None, None


def load_model(ckpt_path, device, override_cfg=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"]) if "cfg" in ck else config.load()
    if override_cfg:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_cfg))
    model = HeightNet(cfg)
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    if "ema" in ck and ck["ema"]:
        # the EMA weights are what validation reported; use them unless asked not to
        for k, v in ck["ema"]["shadow"].items():
            if k in sd:
                sd[k] = v
        model.load_state_dict(sd, strict=False)
        print("  loaded EMA weights")
    return model.to(device).eval(), cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/predictions")
    ap.add_argument("--gsd", type=float, default=None,
                    help="metres/pixel; required for non-georeferenced input")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=float, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-level", action="store_true")
    ap.add_argument("--sun", nargs=2, type=float, default=None, metavar=("ELEV", "AZIM"))
    a = ap.parse_args(argv)

    device = pick_device()
    model, cfg = load_model(a.ckpt, device)
    rgb, gsd_file, crs, transform = read_image(a.image)
    gsd = a.gsd or gsd_file
    if gsd is None:
        sys.exit("no GSD: pass --gsd (metres/pixel). Without it the output cannot be metric.")
    if a.gsd and gsd_file and abs(a.gsd - gsd_file) > 1e-6:
        print(f"  note: --gsd {a.gsd} overrides the file's {gsd_file}")

    print(f"scene {rgb.shape[0]}x{rgb.shape[1]}  gsd={gsd} m/px  device={device}"
          f"  georeferenced={crs is not None}")
    res = predict_scene(model, rgb, gsd,
                        tile=a.tile or int(cfg.infer.tile),
                        overlap=a.overlap if a.overlap is not None else float(cfg.infer.overlap),
                        batch=a.batch, tta=a.tta, level=not a.no_level, device=device)

    stem = Path(a.image).stem
    meta = write_outputs(a.out, stem, res.agl, sigma=res.sigma if res.sigma is not None
                         else res.overlap_disagreement,
                         gsd=gsd, crs=crs, transform=transform,
                         sun={"elevation_deg": a.sun[0], "azimuth_deg": a.sun[1],
                              "source": "cli"} if a.sun else None,
                         model_info={"model": "HeightNet", "checkpoint": str(a.ckpt),
                                     "backbone": str(cfg.model.checkpoint),
                                     "head": str(cfg.model.head)},
                         infer_meta=res.meta,
                         assumptions=[f"GSD taken as {gsd} m/px"
                                      + ("" if gsd_file else " (supplied on the command line)")])
    v = meta["value_range"]
    print(f"\nwrote {a.out}/{stem}.{{tif,json,_preview.png}}")
    print(f"  AGL  p1={v['p1']:.2f}  p50={v['p50']:.2f}  p99={v['p99']:.2f}  max={v['max']:.2f} m")
    print(f"  {meta['quantity_note']}")


if __name__ == "__main__":
    main()
