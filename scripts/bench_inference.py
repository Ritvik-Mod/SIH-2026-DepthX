#!/usr/bin/env python
"""Measure INFERENCE time per scene.  The numbers in hpc/04_probe.pbs are training
throughput -- net.train() with an optimiser step -- so they include a backward pass and
understate inference by roughly 3x.  Quoting them for a demo would be wrong.

Weights are irrelevant to timing, so --ckpt is optional: without it the backbone is
built from the pretrained checkpoint and the heads are randomly initialised, which runs
at exactly the same speed.
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

from heightmap import config
from heightmap.models.net import HeightNet
from heightmap.infer import predict_scene
from heightmap.utils.misc import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="optional; timing does not depend on weights")
    ap.add_argument("--backbone", default="depth-anything/Depth-Anything-V2-Large-hf")
    ap.add_argument("--sizes", nargs="*", type=int, default=[1024, 2048])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gsd", type=float, default=0.33)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    if a.ckpt:
        from heightmap.predict import load_model
        model, cfg = load_model(a.ckpt, dev)
        name = cfg.model.checkpoint.split("/")[-1]
    else:
        cfg = config.load(overrides=[f"model.checkpoint={a.backbone}"])
        model = HeightNet(cfg).to(dev).eval()
        name = a.backbone.split("/")[-1] + " (untrained heads; timing only)"

    tile = int(cfg.data.crop); ov = float(cfg.infer.overlap)
    stride = max(14, tile - int(tile * ov))
    print(f"device {dev}  backbone {name}")
    print(f"tile {tile}  overlap {ov}  stride {stride}  batch {a.batch}  tta {a.tta}\n")

    rows = []
    for S in a.sizes:
        img = (np.random.rand(S, S, 3) * 255).astype(np.uint8)
        n_ax = max(1, -(-(S - tile) // stride) + 1) if S > tile else 1
        ntiles = n_ax * n_ax + (1 if S > tile else 0)      # + one coarse levelling pass
        # warm up: first call pays lazy init, cudnn autotune and allocator growth
        predict_scene(model, img, a.gsd, tile=tile, overlap=ov, batch=a.batch,
                      tta=a.tta, device=dev, progress=False)
        ts = []
        for _ in range(a.repeat):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_scene(model, img, a.gsd, tile=tile, overlap=ov, batch=a.batch,
                          tta=a.tta, device=dev, progress=False)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        mpx = S * S / 1e6
        rows.append(dict(size=S, seconds=round(t, 3), spread=round(max(ts) - min(ts), 3),
                         megapixels=round(mpx, 2), mpx_per_s=round(mpx / t, 3),
                         tiles=ntiles, ms_per_tile=round(1000 * t / max(ntiles, 1), 1)))
        r = rows[-1]
        print(f"{S}x{S}  {r['megapixels']:.2f} Mpx  ->  {r['seconds']:.2f} s "
              f"(+-{r['spread']:.2f})   {r['mpx_per_s']:.2f} Mpx/s   "
              f"~{r['tiles']} forward passes, {r['ms_per_tile']:.0f} ms each")

    if rows:
        mp = float(np.median([r["mpx_per_s"] for r in rows]))
        gsd = a.gsd
        print(f"\nmedian throughput {mp:.2f} Mpx/s at {gsd} m/px")
        for km in (1.0, 10.0, 100.0):
            px = km * 1e6 / (gsd ** 2)          # square metres -> pixels
            print(f"  {km:>5.0f} km2 of {gsd} m/px imagery = {px/1e6:8.1f} Mpx "
                  f"-> {px/1e6/mp/60:7.1f} min")
        print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
