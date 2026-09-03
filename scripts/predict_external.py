#!/usr/bin/env python
"""Run the height model on arbitrary external imagery and package it for Three.js.

For imagery that is NOT GAMUS: drone shots, web aerials, anything with no ground truth.

The critical input is --source-gsd, the metres-per-pixel of YOUR image.  The model
conditions on log(GSD) through FiLM and was trained over 0.165-0.65 m/px (crop 518,
scale 0.5-1.97 about a 0.33 m/px base).  A drone frame is typically 0.05-0.20 m/px, so
feeding it in untouched puts it outside the trained band: the model sees a building
spanning several times more pixels than it expects and scales the height to match.
Each image is therefore RESAMPLED so its effective GSD lands on --target-gsd before
inference.

There is no ground truth here, so nothing is scored.  Heights are conditional on the
GSD you declare -- get that wrong and every metre is wrong by the same factor.
"""
import sys, os, json, argparse, glob
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE)); sys.path.insert(0, _HERE)
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from heightmap.predict import load_model
from heightmap.infer import predict_scene
from heightmap.utils.misc import pick_device
import export_for_threejs as E

Image.MAX_IMAGE_PIXELS = None
TRAINED_LO, TRAINED_HI = 0.165, 0.65      # m/px the model actually saw


def resample_to_gsd(img: np.ndarray, src_gsd: float, dst_gsd: float, max_px: int):
    """Rescale so one output pixel covers dst_gsd metres, then cap the long side."""
    f = src_gsd / dst_gsd                              # <1 shrinks when src is finer
    h, w = img.shape[:2]
    nh, nw = max(64, int(round(h * f))), max(64, int(round(w * f)))
    if max(nh, nw) > max_px:                           # cap for memory/time only
        s = max_px / max(nh, nw)
        nh, nw = max(64, int(nh * s)), max(64, int(nw * s))
        dst_gsd = src_gsd * (h / nh)                   # the cap changes the true GSD
    out = np.asarray(Image.fromarray(img).resize(
        (nw, nh), Image.LANCZOS if f < 1 else Image.BICUBIC), np.uint8)
    return out, dst_gsd, f


def panel(name, rgb, agl, gsd, out_path):
    v = agl[np.isfinite(agl)]
    vmax = max(float(np.percentile(v, 99)) if v.size else 1.0, 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6)); fig.patch.set_facecolor("white")
    ax[0].imshow(rgb); ax[0].set_title("RGB input (resampled)", fontsize=12, fontweight="bold")
    im = ax[1].imshow(np.nan_to_num(agl), cmap="viridis", vmin=0, vmax=vmax)
    ax[1].set_title("Predicted height above ground", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03).set_label("metres AGL", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    H, W = agl.shape
    fig.suptitle(f"{name}   ·   NO GROUND TRUTH — nothing is scored here\n"
                 f"{W}x{H} px at an ASSUMED {gsd:.3f} m/px = {W*gsd:.0f} x {H*gsd:.0f} m"
                 f"   ·   height p50 {np.percentile(v,50):.1f}  p99 {np.percentile(v,99):.1f}"
                 f"  max {v.max():.1f} m",
                 fontsize=11, y=0.99, linespacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.subplots_adjust(top=0.80, wspace=0.05)
    fig.savefig(out_path, dpi=105, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("images", nargs="+", help="image files or a directory")
    ap.add_argument("--source-gsd", type=float, required=True,
                    help="metres per pixel of YOUR images. Estimate it: a car is ~4.5 m "
                         "long, a road lane ~3.5 m wide. gsd = metres / pixels.")
    ap.add_argument("--target-gsd", type=float, default=0.33)
    ap.add_argument("--max-px", type=int, default=2048, help="cap on the long side")
    ap.add_argument("--out", default="outputs/india_export")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    files = []
    for p in a.images:
        files += sorted(glob.glob(os.path.join(p, "*"))) if os.path.isdir(p) else [p]
    files = [f for f in files if os.path.splitext(f)[1].lower()
             in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")]
    if not files:
        sys.exit("no images found")

    if not (TRAINED_LO <= a.target_gsd <= TRAINED_HI):
        print(f"WARNING: target-gsd {a.target_gsd} is outside the trained band "
              f"{TRAINED_LO}-{TRAINED_HI} m/px; heights will be extrapolated.\n")

    import torch
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    print(f"device {dev}  backbone {cfg.model.checkpoint.split('/')[-1]}  images {len(files)}")
    print(f"ASSUMED source GSD {a.source_gsd} m/px  ->  resampled to {a.target_gsd} m/px")
    print(f"heights scale with this assumption. If it is wrong by 2x, so is every metre.\n")

    os.makedirs(a.out, exist_ok=True)
    manifest = []
    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        raw = np.asarray(Image.open(f).convert("RGB"), np.uint8)
        rgb, eff_gsd, factor = resample_to_gsd(raw, a.source_gsd, a.target_gsd, a.max_px)
        agl = predict_scene(model, rgb, eff_gsd, tile=int(cfg.data.crop),
                            overlap=float(cfg.infer.overlap), batch=a.batch,
                            device=dev, progress=False).agl
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        npy = os.path.join(d, f"{name}_agl.npy"); png = os.path.join(d, f"{name}_rgb.png")
        np.save(npy, agl.astype(np.float32)); Image.fromarray(rgb).save(png)
        meta = E.export(npy, png, d, eff_gsd)
        meta.update(field_source="model_prediction", has_ground_truth=False,
                    accuracy_vs_lidar=None,
                    accuracy_note="external imagery, no LiDAR reference: nothing is scored",
                    source_file=os.path.basename(f),
                    original_resolution={"width": int(raw.shape[1]), "height": int(raw.shape[0])},
                    assumed_source_gsd_m=a.source_gsd, effective_gsd_m=eff_gsd,
                    resample_factor=round(factor, 4),
                    gsd_caveat="Heights are proportional to the ASSUMED source GSD. "
                               "The model was trained on 0.165-0.65 m/px; a value outside "
                               "that band is extrapolation.")
        json.dump(meta, open(os.path.join(d, "metadata.json"), "w"), indent=2)
        panel(name, rgb, agl, eff_gsd, os.path.join(d, f"{name}_panel.png"))
        v = agl[np.isfinite(agl)]
        manifest.append(dict(name=name, dir=name, source_file=os.path.basename(f),
                             original=[int(raw.shape[1]), int(raw.shape[0])],
                             processed=[int(rgb.shape[1]), int(rgb.shape[0])],
                             effective_gsd_m=eff_gsd,
                             p50_m=float(np.percentile(v, 50)),
                             p99_m=float(np.percentile(v, 99)), max_m=float(v.max())))
        print(f"  [{i}/{len(files)}] {name:28s} {raw.shape[1]}x{raw.shape[0]} -> "
              f"{rgb.shape[1]}x{rgb.shape[0]} @ {eff_gsd:.3f} m/px   "
              f"p50 {np.percentile(v,50):5.1f}  p99 {np.percentile(v,99):6.1f}  "
              f"max {v.max():6.1f} m")

    json.dump(dict(count=len(manifest), quantity="AGL", has_ground_truth=False,
                   assumed_source_gsd_m=a.source_gsd, target_gsd_m=a.target_gsd,
                   trained_gsd_band=[TRAINED_LO, TRAINED_HI],
                   note="External imagery. No ground truth, nothing scored. Heights are "
                        "conditional on the assumed source GSD.",
                   images=manifest),
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"\n{len(manifest)} images -> {a.out}/")


if __name__ == "__main__":
    main()
