#!/usr/bin/env python
"""Does the DEPLOYED inference path blur more than a single forward pass?

The sharpening experiments evaluate one 518 crop straight out of model(x, gsd).  What
actually ships is predict_scene: a coarse whole-scene levelling pass, then 518 tiles at
25% overlap blended with a ramp window.  Averaging overlapping predictions is a
smoothing operation, and the export .tif is produced that way -- so the field Divyanshu
receives may be blurrier than any number measured so far.

This compares them on identical tiles against the same ground truth, and isolates which
component costs what by sweeping overlap and the levelling pass.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, h5py, torch

from _sharpen_common import evaluate, row_str, header
from heightmap.data.gamus import index_tiles, IMAGE_KINDS
from heightmap.infer import predict_scene
from heightmap.predict import load_model
from heightmap.utils.misc import pick_device


def load_full(root, split, tid):
    with h5py.File(f"{root}/heights/{split}/{tid}_AGL.h5") as f:
        agl = np.asarray(f["image"][()], np.float32)
    with h5py.File(f"{root}/classes/{split}/{tid}_CLS.h5") as f:
        cls = np.asarray(f["image"][()])
    for k in IMAGE_KINDS:
        p = f"{root}/images/{split}/{tid}_{k}.h5"
        if os.path.exists(p):
            with h5py.File(p) as f:
                return np.asarray(f["image"][()]), agl, cls
    return None, agl, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gsd", type=float, default=0.33)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    tile = int(cfg.data.crop)
    ids = [t for _c, t in index_tiles(a.root, a.split)][:a.n]

    data = []
    for tid in ids:
        rgb, gt, cls = load_full(a.root, a.split, tid)
        if rgb is not None:
            data.append((tid, rgb, gt, cls))
    print(f"{len(data)} full tiles from {a.split}  ({data[0][2].shape[0]} px)  "
          f"model tile {tile}\n")

    # single centre crop: the configuration every sharpening number so far was measured on
    @torch.no_grad()
    def crop_pass():
        from heightmap.data.transforms import normalise
        P, G, M, C = [], [], [], []
        for tid, rgb, gt, cls in data:
            H = gt.shape[0]; o = (H - tile) // 2
            r, g, c = rgb[o:o+tile, o:o+tile], gt[o:o+tile, o:o+tile], cls[o:o+tile, o:o+tile]
            x = torch.from_numpy(np.ascontiguousarray(
                normalise(r).transpose(2, 0, 1)))[None].to(dev)
            gs = torch.tensor([a.gsd], device=dev)
            out = model(x, gs if bool(cfg.model.use_gsd) else None)
            P.append(out["height"][0, 0].float().cpu().numpy()); G.append(g)
            M.append(np.isfinite(g) & (g > -100.0)); C.append(c)
        return P, G, M, C

    def tiled_pass(overlap, level, match_crop=True):
        """match_crop keeps the comparison honest.  predict_scene runs on the full 1024
        tile, but the single-crop row scores only the centre 518 -- 25% of the area, a
        different set of buildings.  Band statistics across those two populations are
        not comparable, so the tiled output is cropped to the same centre window before
        scoring.  Pass match_crop=False to score the whole tile instead."""
        P, G, M, C = [], [], [], []
        for tid, rgb, gt, cls in data:
            r = predict_scene(model, rgb, a.gsd, tile=tile, overlap=overlap, batch=4,
                              level=level, device=dev, progress=False).agl
            g, c = gt, cls
            if match_crop:
                H = gt.shape[0]; o = (H - tile) // 2
                r = r[o:o+tile, o:o+tile]; g = gt[o:o+tile, o:o+tile]
                c = cls[o:o+tile, o:o+tile]
            P.append(r); G.append(g)
            M.append(np.isfinite(g) & (g > -100.0)); C.append(c)
        return P, G, M, C

    print(header())
    print("MATCHED: every row scores the SAME centre 518 window, so band statistics "
          "are comparable.")
    print(row_str("single 518 crop", evaluate(*crop_pass())))
    for ov in (0.0, 0.25, 0.5):
        for lv in (True, False):
            tag = f"tiled ov={ov:g} level={'on' if lv else 'off'}"
            print(row_str(tag, evaluate(*tiled_pass(ov, lv, True))))
    print()
    print("UNMATCHED: tiled rows scored over the full 1024 tile.  Compare these only "
          "with each other,\nnever with the 518 crop above -- different pixels, "
          "different buildings.")
    for ov in (0.0, 0.25):
        for lv in (True, False):
            tag = f"full-tile ov={ov:g} level={'on' if lv else 'off'}"
            print(row_str(tag, evaluate(*tiled_pass(ov, lv, False))))
    print("-" * 118)
    print("\nIf 'single 518 crop' is much sharper than the tiled rows, the blending\n"
          "window is smoothing the deliverable and the fix is in infer.py, not in any\n"
          "post-process.  If overlap=0 recovers it, the ramp window is the cause.\n"
          "If levelling off recovers it, the coarse guide is.  If all rows match, the\n"
          "deployed path is not the problem and the export .tif is as sharp as the\n"
          "model gets.")


if __name__ == "__main__":
    main()
