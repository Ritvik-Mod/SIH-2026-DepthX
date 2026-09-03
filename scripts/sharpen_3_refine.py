#!/usr/bin/env python
"""IDEA 3 -- a learned refinement network.

A small residual CNN at FULL resolution:

    [coarse height (1), RGB (3), semantic probs (7)]  ->  4-5 convs  ->  height residual

Why a residual and not a fresh prediction: absolute height regression is hard, whereas
correcting a good coarse estimate given full-resolution RGB is easy, and RGB carries the
edge LOCATION exactly.  Same argument as cascaded refinement in segmentation.

Why this can beat the base model at its own gradient loss: the heads emit at 296x296 and
bilinearly upsample to 518 (heads.py:63, :111), so the finest structure the base model
can express is ~1.75 output pixels wide.  The gradient loss has been scoring sharpness at
518 for sixty epochs against an architecture that cannot produce it -- sharpness sat at
0.085 / 0.085 / 0.084 across A1, A2 and A6.  This head runs at 518 and has no such
ceiling.

The base model is FROZEN and runs under no_grad.  Only the refiner trains, so this is
cheap despite operating at full resolution.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from _sharpen_common import build, evaluate, row_str, header
from heightmap import config
from heightmap.data.gamus import GamusDataset, index_tiles
from heightmap.data.classes import BUILDING
from heightmap.data.transforms import denormalise
from heightmap.losses import terms
from heightmap.metrics import sharpness
from heightmap.predict import load_model
from heightmap.utils.misc import pick_device


class Refiner(nn.Module):
    """Zero-initialised residual: at step 0 the output is exactly the base prediction,
    so training can only improve on it and a failed run degrades gracefully."""

    def __init__(self, in_ch=11, ch=48, depth=4, h_max=250.0):
        super().__init__()
        L, c = [], in_ch
        for i in range(depth):
            d = 1 if i % 2 == 0 else 2
            # padding MUST equal the dilation for a 3x3 kernel, or the output shrinks by
            # 2*(d-1) per layer and the residual no longer lines up with `coarse`.
            L += [nn.Conv2d(c, ch, 3, padding=d, dilation=d,
                            padding_mode="replicate"), nn.GroupNorm(8, ch), nn.GELU()]
            c = ch
        self.body = nn.Sequential(*L)
        self.out = nn.Conv2d(ch, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.h_max = h_max

    def forward(self, coarse, rgb01, sem_prob):
        z = torch.cat([coarse / 50.0, rgb01, sem_prob], 1)
        return (coarse + self.out(self.body(z))).clamp(0.0, self.h_max)


@torch.no_grad()
def base_forward(model, batch, dev, use_gsd):
    x = batch["image"].to(dev, non_blocking=True)
    g = batch["gsd"].to(dev, non_blocking=True)
    out = model(x, g if use_gsd else None)
    sem = out["semantic"].softmax(1) if "semantic" in out else \
        torch.zeros(x.shape[0], 7, *x.shape[-2:], device=dev)
    return out["height"].float(), denormalise(batch["image"]).to(dev), sem.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-grad", type=float, default=2.0,
                    help="gradient-loss weight. Higher than the base recipe on purpose: "
                         "sharpness is the whole point of this head.")
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--ch", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out", default="outputs/refiner")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    cfg.data.root = a.root; cfg.data.sun_cache = ""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    use_gsd = bool(cfg.model.use_gsd)

    tr_ids = [t for _c, t in index_tiles(a.root, a.train_split)][:a.n_train]
    va_ids = [t for _c, t in index_tiles(a.root, a.val_split)][:a.n_val]
    tr = GamusDataset(cfg, a.train_split, True, [(t.split("_")[0], t) for t in tr_ids])
    va = GamusDataset(cfg, a.val_split, False, [(t.split("_")[0], t) for t in va_ids])
    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=6,
                    persistent_workers=True, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=4)

    net = Refiner(11, a.ch, a.depth, float(cfg.model.h_max)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * max(len(dl), 1))
    print(f"device {dev}   refiner params {sum(p.numel() for p in net.parameters()):,}"
          f"   train {len(tr)}  val {len(va)}\n")

    def run_val():
        net.eval(); P, G, M, C = [], [], [], []
        with torch.no_grad():
            for b in vl:
                coarse, rgb, sem = base_forward(model, b, dev, use_gsd)
                ref = net(coarse, rgb, sem)
                for k in range(coarse.shape[0]):
                    P.append(ref[k, 0].cpu().numpy()); G.append(b["height"][k, 0].numpy())
                    M.append(b["mask"][k, 0].numpy().astype(bool)); C.append(b["cls"][k].numpy())
        net.train(); return P, G, M, C

    def base_val():
        P, G, M, C = [], [], [], []
        with torch.no_grad():
            for b in vl:
                coarse, _, _ = base_forward(model, b, dev, use_gsd)
                for k in range(coarse.shape[0]):
                    P.append(coarse[k, 0].cpu().numpy()); G.append(b["height"][k, 0].numpy())
                    M.append(b["mask"][k, 0].numpy().astype(bool)); C.append(b["cls"][k].numpy())
        return P, G, M, C

    print(header())
    base = evaluate(*base_val())
    print(row_str("baseline (frozen model)", base))

    os.makedirs(a.out, exist_ok=True)
    net.train()
    for ep in range(a.epochs):
        t0 = time.time()
        for b in dl:
            coarse, rgb, sem = base_forward(model, b, dev, use_gsd)
            gt = b["height"].to(dev); m = b["mask"].to(dev)
            ref = net(coarse, rgb, sem)
            l1 = terms.l1_loss(ref, gt, m)
            gr = terms.multiscale_gradient_loss(ref, gt, m, int(cfg.loss.grad_scales))
            loss = a.w_l1 * l1 + a.w_grad * gr
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
        res = evaluate(*run_val())
        print(row_str(f"refined epoch {ep} ({time.time()-t0:.0f}s)", res))
        torch.save({"model": net.state_dict(), "in_ch": 11, "ch": a.ch,
                    "depth": a.depth, "h_max": float(cfg.model.h_max),
                    "base_ckpt": a.ckpt, "val": res},
                   os.path.join(a.out, "refiner.pt"))
    print("-" * 118)
    print(f"\nrefiner.pt -> {a.out}/refiner.pt")
    print("The residual is zero-initialised, so epoch -1 is exactly the baseline.  If\n"
          "sharpness does not move while L1 falls, the head is fixing magnitude rather\n"
          "than edges: raise --w-grad.  If sharpness moves and building MAE worsens, it\n"
          "is inventing edges: lower it.")


if __name__ == "__main__":
    main()
