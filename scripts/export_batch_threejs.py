#!/usr/bin/env python
"""Run inference on a set of GAMUS tiles and package each one for Three.js.

Produces out/<tile_id>/ containing the four files export_for_threejs.py writes, plus
the raw prediction as .npy and the source RGB, so downstream work never has to re-run
the model to get numbers.

The tile list comes from a demo run's compare.json by default, so the tiles exported
are exactly the ones that were selected -- retyping twelve ids is how the wrong tile
ends up in a deliverable.

The model is loaded ONCE.  Calling export_for_threejs.py twelve times from a shell loop
would rebuild a ViT-L from the HuggingFace cache twelve times over.
"""
import sys, os, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import numpy as np, h5py
from PIL import Image

from heightmap.data.gamus import index_tiles, IMAGE_KINDS
from heightmap.data.classes import BUILDING
from heightmap.predict import load_model
from heightmap.infer import predict_scene
from heightmap.utils.misc import pick_device
import export_for_threejs as E


def load_tile(root, split, tid):
    with h5py.File(f"{root}/heights/{split}/{tid}_AGL.h5") as f:
        agl = np.asarray(f["image"][()], np.float32)
    with h5py.File(f"{root}/classes/{split}/{tid}_CLS.h5") as f:
        cls = np.asarray(f["image"][()])
    for kind in IMAGE_KINDS:                      # DC/PHL _RGB, NYC _IMG
        p = f"{root}/images/{split}/{tid}_{kind}.h5"
        if os.path.exists(p):
            with h5py.File(p) as f:
                return np.asarray(f["image"][()]), agl, cls
    return None, agl, cls


def tile_stats(pred, gt, cls):
    finite = np.isfinite(gt) & (gt > -100.0)
    out = {}
    for name, m in (("overall", finite), ("building", finite & (cls == BUILDING))):
        p, g = pred[m], gt[m]
        if p.size < 16:
            out[name] = dict(n=int(p.size), mae=None, bias=None, r=None); continue
        r = None
        if p.std() > 1e-6 and g.std() > 1e-6:
            r = float(np.corrcoef(p, g)[0, 1])
        out[name] = dict(n=int(p.size), mae=float(np.abs(p - g).mean()),
                         bias=float((p - g).mean()), r=r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="required only when --source pred")
    ap.add_argument("--source", default="pred", choices=["pred", "gt"],
                    help="pred = run the model.  gt = package the LiDAR ground truth "
                         "through the identical path, which isolates whether a problem "
                         "is in the model output or in the export/render chain.")
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="train")
    ap.add_argument("--from-json", default="",
                    help="a demo run's compare.json; its 'tile' fields become the list")
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--out", default="outputs/threejs_export")
    ap.add_argument("--gsd", type=float, default=E.DEFAULT_GSD)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    tids = list(a.tiles)
    if a.from_json:
        rows = json.load(open(a.from_json))
        tids = [r["tile"] for r in rows] + [t for t in tids if t not in
                                            {r["tile"] for r in rows}]
    if not tids:
        sys.exit("no tiles: pass --from-json <compare.json> or --tiles <id> ...")

    model = cfg = dev = None
    if a.source == "pred":
        if not a.ckpt:
            sys.exit("--source pred needs --ckpt")
        import torch
        dev = pick_device() if a.device == "auto" else torch.device(a.device)
        model, cfg = load_model(a.ckpt, dev)
        print(f"device {dev}  backbone {cfg.model.checkpoint.split('/')[-1]}  "
              f"tiles {len(tids)}  split {a.split}  source PREDICTION\n")
    else:
        print(f"tiles {len(tids)}  split {a.split}  source GROUND TRUTH "
              f"(no model loaded, no inference)\n")

    root_out = os.path.abspath(a.out)
    os.makedirs(root_out, exist_ok=True)
    manifest = []
    for i, tid in enumerate(tids, 1):
        rgb, gt, cls = load_tile(a.root, a.split, tid)
        if rgb is None:
            print(f"  [{i}/{len(tids)}] skip {tid}: no image found"); continue
        if a.source == "pred":
            field = predict_scene(model, rgb, a.gsd, tile=int(cfg.data.crop),
                                  overlap=float(cfg.infer.overlap), batch=a.batch,
                                  device=dev, progress=False).agl
        else:
            # GT carries non-finite pixels and a handful of negatives; export() maps
            # both to the nodata sentinel, so pass it through untouched rather than
            # cleaning it here.  Cleaning would hide exactly the artefacts this run is
            # meant to expose.
            field = gt.astype(np.float32)
        d = os.path.join(root_out, tid)
        os.makedirs(d, exist_ok=True)
        npy = os.path.join(d, f"{tid}_agl.npy")
        png = os.path.join(d, f"{tid}_rgb.png")
        np.save(npy, field.astype(np.float32))
        Image.fromarray(rgb.astype(np.uint8)).save(png)
        meta = E.export(npy, png, d, a.gsd)
        meta["split"] = a.split
        # Stamp provenance hard.  A GT heightmap and a predicted one are byte-shaped
        # identically, and a folder of LiDAR truth mistaken for model output would make
        # the model look perfect.
        meta["field_source"] = ("model_prediction" if a.source == "pred"
                                else "ground_truth_lidar")
        if a.source == "pred":
            st = tile_stats(field, gt, cls)
            meta["accuracy_vs_lidar"] = st
        else:
            st = {"overall": {"mae": 0.0, "bias": 0.0, "r": 1.0},
                  "building": {"mae": 0.0, "bias": 0.0, "r": 1.0}}
            meta["accuracy_vs_lidar"] = None
            meta["accuracy_note"] = ("this IS the ground truth; there is nothing to "
                                     "score it against. Not a model output.")
        with open(os.path.join(d, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        manifest.append(dict(tile=tid, dir=tid, split=a.split,
                             field_source=meta["field_source"],
                             min_m=meta["min_height_m"], max_m=meta["max_height_m"],
                             mean_m=meta["mean_height_m"],
                             building_mae=st["building"]["mae"],
                             building_r=st["building"]["r"]))
        b = st["building"]
        tag = "" if a.source == "pred" else "  [GROUND TRUTH]"
        print(f"  [{i}/{len(tids)}] {tid:14s} range {meta['min_height_m']:6.2f}.."
              f"{meta['max_height_m']:7.2f} m  bldg MAE "
              f"{b['mae'] if b['mae'] is None else round(b['mae'],3)}{tag}  -> {tid}/")

    with open(os.path.join(root_out, "manifest.json"), "w") as f:
        json.dump(dict(count=len(manifest), split=a.split,
                       field_source=("model_prediction" if a.source == "pred"
                                     else "ground_truth_lidar"),
                       pixel_spacing_m=a.gsd, quantity="AGL",
                       quantity_note="height above LOCAL GROUND. DSM = DTM + heightmap. "
                                     "ADD the terrain, do not subtract.",
                       tiles=manifest), f, indent=2)
    print(f"\n{len(manifest)} tiles -> {root_out}/")
    print(f"  manifest.json lists every tile with its height range and accuracy")
    print(f"  each <tile>/ has heightmap.tif (float32 metres), texture.png,")
    print(f"  heightmap_preview.png, metadata.json, plus the raw .npy and RGB")


if __name__ == "__main__":
    main()
