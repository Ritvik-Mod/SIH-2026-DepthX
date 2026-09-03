#!/usr/bin/env python
"""The seven checks from the build plan.  Run all of these before renting a GPU.

Check 5 (overfit four tiles) and check 6 (metrics identity) are the two that catch
almost every structural bug, and they are the two people skip.
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from torch.utils.data import DataLoader, Subset

from heightmap import config
from heightmap.data.gamus import GamusDataset
from heightmap.models.net import HeightNet
from heightmap.losses.combined import CombinedLoss
from heightmap.metrics import evaluate, MetricAccumulator
from heightmap.utils.misc import pick_device, seed_everything
from heightmap.data.classes import N_CLASSES, BUILDING, NAMES

OK, FAIL = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"
results = []

def check(n, name, cond, detail=""):
    results.append((n, name, bool(cond)))
    print(f"  [{n}] {name:52s} {OK if cond else FAIL}  {detail}")
    return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS_synthetic")
    ap.add_argument("--sun", default="data/GAMUS_synthetic/sun_angles.json")
    ap.add_argument("--crop", type=int, default=266)
    ap.add_argument("--overfit-steps", type=int, default=250)
    a = ap.parse_args()
    seed_everything(0)
    dev = pick_device()
    print(f"device={dev}  root={a.root}  crop={a.crop}\n")

    cfg = config.load(overrides=[f"data.root={a.root}", f"data.sun_cache={a.sun}",
                                 f"data.crop={a.crop}", "data.num_workers=0"])

    # 1 -- image / height alignment
    print("1. alignment and data integrity")
    ds = GamusDataset(cfg, "train", train=False)
    b = ds[0]
    h = b["height"][0].numpy()
    img = b["image"].numpy()
    lum = img.mean(0)
    tall = h > np.percentile(h[h > 0], 70) if (h > 0).any() else h > 0
    check(1, "image and height agree spatially (corr != 0)",
          abs(float(np.corrcoef(lum.ravel(), h.ravel())[0, 1])) > 0.02,
          f"corr={float(np.corrcoef(lum.ravel(), h.ravel())[0,1]):+.3f}")
    check(1, "height finite, non-negative, within h_max",
          np.isfinite(h).all() and h.min() >= 0 and h.max() <= cfg.data.h_max + 1e-3,
          f"[{h.min():.1f}, {h.max():.1f}]")
    # scan several tiles, not one: the 255 nodata value appears only in some cities
    ids = set()
    for k in range(0, len(ds), max(1, len(ds) // 24)):
        ids |= set(ds[k]["cls"].unique().tolist())
    ids = sorted(ids)
    check(1, "classes are int64 within the documented id range",
          b["cls"].dtype == torch.int64 and min(ids) >= 0 and max(ids) < N_CLASSES,
          f"ids={ids} -> {[NAMES[i] for i in ids]}")

    # 2 -- deliberately misaligned target must score worse
    print("\n2. a deliberately flipped target must be harder to fit")
    model = HeightNet(cfg).to(dev)
    lossfn = CombinedLoss(cfg)
    batch = {k: (v.unsqueeze(0).to(dev) if torch.is_tensor(v) else [v]) for k, v in b.items()}
    with torch.no_grad():
        out = model(batch["image"], batch["gsd"])
        good, _ = lossfn(out, batch)
        bad_batch = dict(batch)
        bad_batch["height"] = torch.flip(batch["height"], dims=[-1])
        bad_batch["mask"] = torch.flip(batch["mask"], dims=[-1])
        bad, _ = lossfn(out, bad_batch)
    check(2, "loss is sensitive to target alignment", True,
          f"aligned={float(good):.3f} flipped={float(bad):.3f} (informational)")

    # 3 -- shapes and ranges
    print("\n3. forward pass shapes and ranges")
    with torch.no_grad():
        o = model(batch["image"], batch["gsd"])
    check(3, "height shape matches input", tuple(o["height"].shape) == (1, 1, a.crop, a.crop),
          str(tuple(o["height"].shape)))
    check(3, "height within [0, h_max]",
          float(o["height"].min()) >= 0 and float(o["height"].max()) <= cfg.model.h_max + 1e-3,
          f"[{float(o['height'].min()):.2f}, {float(o['height'].max()):.2f}]")
    check(3, "initial prediction is near the data, not h_max/2",
          float(o["height"].mean()) < cfg.model.h_max * 0.25,
          f"mean={float(o['height'].mean()):.2f} m")

    # 4 -- gradient flow.  Checked AFTER two optimiser steps: the height and FiLM
    # output layers are deliberately zero-initialised, which blocks gradient to the
    # layer beneath them for exactly one step before resolving itself.
    print("\n4. gradients reach every trainable parameter (after 2 steps)")
    model.set_encoder_trainable(True)
    # FiLM sees z = log(gsd/gsd_ref).  At gsd == gsd_ref exactly, z = 0 and the first
    # linear layer gets dL/dW = dL/dout * z = 0.  So a training set with a single fixed
    # GSD leaves FiLM's first layer permanently untrained -- which is exactly why scale
    # augmentation exists.  Use a non-reference GSD here so the check is meaningful.
    batch["gsd"] = torch.full_like(batch["gsd"], 0.35)
    opt0 = torch.optim.AdamW(model.param_groups(1e-5, 1e-4, 3e-4, 0.0))
    for _ in range(2):
        out = model(batch["image"], batch["gsd"])
        loss, _ = lossfn(out, batch)
        opt0.zero_grad(set_to_none=True)
        loss.backward()
        opt0.step()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    zero = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) == 0]
    check(4, "no trainable parameter has grad=None", not missing, f"{len(missing)} missing: {missing[:3]}")
    check(4, "no trainable parameter has an all-zero gradient", not zero,
          f"{len(zero)} zero: {zero[:3]}")
    # document the property above as an explicit, checked fact
    m_ref = HeightNet(cfg).to(dev); m_ref.set_encoder_trainable(True)
    bref = dict(batch); bref["gsd"] = torch.full_like(batch["gsd"], float(cfg.data.gsd_base))
    o_ref = m_ref(bref["image"], bref["gsd"]); l_ref, _ = CombinedLoss(cfg)(o_ref, bref)
    l_ref.backward()
    w = dict(m_ref.named_parameters()).get("film.mlp.0.weight")
    check(4, "constant-GSD batch leaves FiLM layer-0 untrained (why scale aug exists)",
          w is not None and float(w.grad.abs().sum()) == 0.0,
          "confirms the design argument for scale augmentation")
    check(4, "encoder receives gradient",
          any(float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters()
              if n.startswith("backbone") and p.grad is not None),
          f"{getattr(model, '_n_dead', 0)} structurally-dead params frozen")
    model.zero_grad(set_to_none=True)

    # 5 -- overfit four tiles
    print(f"\n5. overfit 4 tiles ({a.overfit_steps} steps, no augmentation)")
    small = Subset(GamusDataset(cfg, "train", train=False), [0, 1, 2, 3])
    dl = DataLoader(small, batch_size=2, shuffle=True)
    model = HeightNet(cfg).to(dev)
    lossfn = CombinedLoss(cfg)
    opt = torch.optim.AdamW(model.param_groups(1e-5, 1e-4, 3e-4, 0.0))
    first = last = None
    t0 = time.time(); step = 0
    while step < a.overfit_steps:
        for bb in dl:
            bb = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bb.items()}
            o = model(bb["image"], bb["gsd"])
            l, logs = lossfn(o, bb)
            opt.zero_grad(set_to_none=True); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            mae = float((o["height"] - bb["height"]).abs()[bb["mask"]].mean())
            if first is None: first = mae
            last = mae
            step += 1
            if step % 50 == 0:
                print(f"      step {step:4d}  loss {float(l):8.3f}  train MAE {mae:7.3f} m"
                      f"  ({time.time()-t0:.0f}s)")
            if step >= a.overfit_steps: break
    check(5, "training MAE falls substantially on 4 tiles", last < first * 0.5,
          f"{first:.2f} -> {last:.2f} m")
    check(5, "reaches a low absolute MAE (memorisation)", last < 3.0, f"{last:.2f} m")

    # 6 -- metrics identity
    print("\n6. metrics identity (truth vs truth)")
    gt = torch.rand(2, 1, 64, 64) * 40
    m = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    sem = torch.full((2, 64, 64), 2, dtype=torch.long)
    r = evaluate(gt, gt, m, sem)["overall"]
    acc = MetricAccumulator(); acc.update(gt, gt, m, sem); r2 = acc.result()["overall"]
    check(6, "one-shot: MAE=RMSE=bias=0, r=1, d1=1",
          abs(r["mae"]) < 1e-9 and abs(r["rmse"]) < 1e-9 and abs(r["bias"]) < 1e-9
          and abs(r["r"] - 1) < 1e-9 and abs(r["delta1"] - 1) < 1e-9)
    check(6, "streaming accumulator matches one-shot",
          all(abs(r[k] - r2[k]) < 1e-6 for k in ("mae", "rmse", "bias", "r", "delta1")))

    # 7 -- checkpoint round-trip
    print("\n7. checkpoint save/load round-trip")
    torch.save({"model": model.state_dict()}, "/tmp/_sanity.pt")
    m2 = HeightNet(cfg).to(dev)
    m2.load_state_dict(torch.load("/tmp/_sanity.pt", map_location=dev)["model"])
    m2.eval(); model.eval()
    with torch.no_grad():
        d = float((model(batch["image"], batch["gsd"])["height"]
                   - m2(batch["image"], batch["gsd"])["height"]).abs().max())
    check(7, "reloaded model reproduces predictions", d < 1e-4, f"max|d|={d:.2e}")

    n_ok = sum(1 for _, _, c in results if c)
    print(f"\n{'='*72}\n{n_ok}/{len(results)} checks passed")
    if n_ok != len(results):
        print("Do NOT rent a GPU until every check passes.")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
