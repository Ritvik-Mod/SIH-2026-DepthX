#!/usr/bin/env python
"""Strip a training checkpoint down to the weights a release actually needs.

A training checkpoint carries model weights + EMA shadow + AdamW's two moment buffers
+ the grad scaler, because a shared-queue job must be able to resume mid-run.  For
ViT-L that is ~5.0 GB for a ~1.34 GB model.  Nothing downstream of training needs the
optimiser state: inference, export and the Three.js bundles all just want weights.

Two reasons to care beyond tidiness:
  * GitHub release assets cap at 2 GB, so a 5 GB checkpoint cannot be attached at all.
  * A 5 GB download for a 1.34 GB model is a 4x tax on everyone who clones the release.

The EMA shadow is MERGED IN rather than shipped alongside, because the EMA weights are
the ones validation reported -- 1.585 m held-out building MAE is an EMA number.
Shipping raw weights and letting the consumer decide would mean the released model
scores differently from the released metrics, which is the kind of quiet mismatch
nobody catches until it is in a report.  heightmap/predict.py:load_model applies the
same merge, so a stripped checkpoint and the original load to identical weights.

RESUMING FROM THE OUTPUT IS NOT POSSIBLE and that is deliberate: there is no optimiser
state to restore.  Keep the original best.pt on the cluster.  Use this only for release
artefacts and for `train.init_from`, which is weights-only by design.
"""
from __future__ import annotations
import argparse, hashlib, os, sys

import torch

KEEP = ("model", "cfg", "epoch", "step", "val", "best")
DROP = ("optim", "scaler", "ema")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="strip a checkpoint for release")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--keep-raw", action="store_true",
                    help="do NOT merge the EMA shadow; ship the raw training weights. "
                         "Only for debugging -- the reported metrics are EMA numbers.")
    a = ap.parse_args()

    if os.path.exists(a.dst):
        sys.exit(f"refusing to overwrite {a.dst}")

    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    src_gb = os.path.getsize(a.src) / 2**30
    print(f"source {a.src}  {src_gb:.2f} GB")
    print(f"  keys: {sorted(ck.keys())}")

    sd = ck["model"]
    merged = 0
    if not a.keep_raw and ck.get("ema"):
        for k, v in ck["ema"]["shadow"].items():
            if k in sd:
                sd[k] = v
                merged += 1
        print(f"  merged {merged} EMA tensors into the weights "
              f"(these are the weights validation reported)")
    elif a.keep_raw:
        print("  --keep-raw: EMA NOT merged; released weights will not match the "
              "reported metrics")
    else:
        print("  no EMA in this checkpoint")

    out = {k: ck[k] for k in KEEP if k in ck}
    out["release_note"] = ("weights only: optimiser state stripped, EMA merged. "
                           "Cannot be resumed from -- use for inference, export, or "
                           "train.init_from.")
    dropped = [k for k in DROP if k in ck]

    tmp = a.dst + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, a.dst)          # atomic, same habit as train.py

    dst_gb = os.path.getsize(a.dst) / 2**30
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"\nwrote {a.dst}")
    print(f"  dropped: {', '.join(dropped) if dropped else 'nothing'}")
    print(f"  {n/1e6:.1f}M parameters")
    print(f"  {src_gb:.2f} GB -> {dst_gb:.2f} GB  ({100*(1-dst_gb/src_gb):.0f}% smaller)")
    print(f"  epoch {out.get('epoch')}  step {out.get('step')}")
    if isinstance(out.get("val"), dict):
        b = out["val"].get("building", {})
        if b.get("mae") is not None:
            print(f"  recorded building MAE {b['mae']:.3f}  r {b.get('r')}")
    print(f"  sha256 {sha256(a.dst)}")
    if dst_gb > 2.0:
        print("\n  WARNING: still over 2 GB -- GitHub release assets cap there.")
    else:
        print(f"\n  fits a GitHub release asset ({dst_gb:.2f} GB < 2 GB).")


if __name__ == "__main__":
    main()
