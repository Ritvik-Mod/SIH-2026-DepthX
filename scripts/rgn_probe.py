#!/usr/bin/env python
"""Per-block relative gradient norm, measured on two domains, to decide WHICH blocks to
fine-tune rather than guessing.

Lee et al., "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts" (ICLR
2023, arXiv 2210.11466) found that the best subset of layers to tune depends on the KIND
of shift -- first blocks for input-level shifts (corruption, sensor, appearance), middle
for feature-level, last for output/label-level -- and that tuning only the first block
can beat full fine-tuning by ~3% on CIFAR-10-C.  They propose relative gradient norm,

    RGN(group) = ||g||_2 / ||theta||_2

as an automatic selection criterion that needs no extra hyperparameters.

What this script adds over a bare RGN reading: RGN is computed on TWO domains and the
RATIO is reported.  An absolute RGN curve is dominated by depth-dependent gradient
scaling and by how large each block's weights happen to be, so it says little on its
own.  RGN_shifted / RGN_reference divides that structure out and leaves the part that
is actually about the domain shift.

IMPORTANT about interpretation: both domains default to tiles the model never trained
on.  If you point domain A at training tiles instead, the ratio conflates "this block
must change for the new domain" with "this block has already memorised the old one",
and the result is not a shift measurement any more.

Gradients are ACCUMULATED across batches and the norm is taken once at the end, so what
is measured is the norm of the mean gradient -- the thing an optimiser step would
actually see -- not the mean of per-batch norms, which is inflated by noise.

Nothing is stepped.  The optimiser does not exist here.  The checkpoint is never
written.
"""
from __future__ import annotations
import sys, os, re, json, argparse
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch
from torch.utils.data import DataLoader

from heightmap.data.gamus import GamusDataset, ConcatSplits, index_tiles
from heightmap.losses.combined import CombinedLoss
from heightmap.predict import load_model
from heightmap.utils.misc import pick_device, amp_dtype_for, seed_everything

BLK = re.compile(r"backbone\.encoder\.layer\.(\d+)\.")
FUS = re.compile(r"neck\.fusion_stage\.layers\.(\d+)\.")
RSM = re.compile(r"neck\.reassemble_stage\.layers\.(\d+)\.")


def group_of(name: str) -> str | None:
    """Map a parameter name onto the unit we would freeze or unfreeze as a whole.

    Verified against the real DAv2 parameter names, not guessed:
      backbone.embeddings.*                  patch embed + cls + pos
      backbone.encoder.layer.{i}.*           the transformer blocks
      neck.reassemble_stage.layers.{i}.*     one per tap, i indexes out_indices
      neck.convs.{i}                         1x1 channel projections
      neck.fusion_stage.layers.{i}.*         coarse-to-fine fusion, 0 = deepest
    """
    m = BLK.search(name)
    if m:
        return f"blk{int(m.group(1)):02d}"
    if name.startswith("backbone.embeddings"):
        return "embed"
    m = RSM.search(name)
    if m:
        return f"neck.reasm{m.group(1)}"
    m = FUS.search(name)
    if m:
        return f"neck.fuse{m.group(1)}"
    if name.startswith("neck.convs"):
        return "neck.convs"
    if name.startswith("backbone."):
        return "backbone.other"
    if name.startswith("neck."):
        return "neck.other"
    for h in ("film", "height_head", "sem_head", "unc_head"):
        if name.startswith(h):
            return f"head.{h}"
    return "other"


def order_key(g: str):
    """Sort so the table reads input -> output."""
    if g == "embed":
        return (0, 0)
    if g.startswith("blk"):
        return (1, int(g[3:]))
    if g.startswith("neck.reasm"):
        return (2, int(g[10:]))
    if g == "neck.convs":
        return (3, 0)
    if g.startswith("neck.fuse"):
        return (4, int(g[9:]))
    if g.startswith("neck"):
        return (5, 0)
    if g.startswith("head"):
        return (6, hash(g) % 100)
    return (7, 0)


def city_tiles(root, cities, splits, limit):
    """Tiles from the named cities, drawn from splits the model did not train on.

    Returns {split: [(city, tile_id)]} because ConcatSplits builds one GamusDataset per
    key and GamusDataset._open reads root/images/<split>/ -- a synthetic key resolves to
    a directory that does not exist.  That was bug #18; do not re-introduce it.
    """
    want = {c.upper() for c in cities}
    out, budget = {}, limit
    for s in splits:
        got = [t for t in index_tiles(root, s) if t[0].upper() in want][:budget]
        if got:
            out[s] = got
            budget -= len(got)
        if budget <= 0:
            break
    if not out:
        raise SystemExit(f"no tiles for cities {sorted(want)} in splits {splits}")
    return out


def accumulate_grads(model, lossfn, loader, device, amp_dt, n_batches, tag):
    """Sum gradients over n_batches without ever stepping.  Returns batches actually run."""
    model.zero_grad(set_to_none=True)
    model.train()                      # gradient path must match training, not eval
    done = 0
    for b in loader:
        if done >= n_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dt) if amp_dt
               else torch.autocast(device_type=device.type, enabled=False))
        with ctx:
            out = model(batch["image"], batch["gsd"])
            loss, _ = lossfn(out, batch)
        # No GradScaler: this is bf16 (or fp32), which does not underflow the way fp16
        # does, and a scaler would multiply every gradient by the same constant anyway
        # -- which cancels in the A/B ratio but not in the absolute column.
        (loss / n_batches).backward()
        done += 1
        print(f"    {tag} batch {done}/{n_batches}  loss {float(loss):.4f}", flush=True)
    if done == 0:
        raise SystemExit(f"{tag}: loader yielded no batches")
    return done


def rgn_table(model):
    """-> {group: {"rgn":…, "gnorm":…, "pnorm":…, "n":…}} from currently-held .grad."""
    acc = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = group_of(n)
        if g is None:
            continue
        d = acc.setdefault(g, {"g2": 0.0, "p2": 0.0, "n": 0})
        d["g2"] += float(p.grad.detach().float().pow(2).sum())
        d["p2"] += float(p.detach().float().pow(2).sum())
        d["n"] += p.numel()
    out = {}
    for g, d in acc.items():
        gn, pn = d["g2"] ** 0.5, d["p2"] ** 0.5
        out[g] = {"rgn": gn / pn if pn > 0 else float("nan"),
                  "gnorm": gn, "pnorm": pn, "n": d["n"]}
    return out


def main():
    ap = argparse.ArgumentParser(description="per-block RGN on two domains")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--ref-cities", nargs="+", default=["DC", "PHL"],
                    help="domain A -- the domain the model is already good at")
    ap.add_argument("--shift-cities", nargs="+", default=["NYC"],
                    help="domain B -- the shifted domain")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="draw from splits the model did NOT train on")
    ap.add_argument("--tiles", type=int, default=128, help="tiles per domain")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default="outputs/rgn_probe.json")
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = pick_device() if a.device == "auto" else torch.device(a.device)
    model, cfg = load_model(a.ckpt, dev)
    amp_dt = amp_dtype_for(dev, str(cfg.train.amp_dtype))
    print(f"device {dev}  amp {amp_dt}  backbone {cfg.model.checkpoint.split('/')[-1]}")

    # Augmentation off on BOTH domains.  With it on, photometric jitter is itself an
    # input-level shift and would show up in the ratio as if it were the city.
    cfg.data.root = a.root
    lossfn = CombinedLoss(cfg)

    res, counts = {}, {}
    for tag, cities in (("ref", a.ref_cities), ("shift", a.shift_cities)):
        tb = city_tiles(a.root, cities, a.splits, a.tiles)
        ds = ConcatSplits(cfg, tb, train=False)
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=2,
                        pin_memory=False)
        print(f"\n[{tag}] cities {cities}  tiles {len(ds)}  "
              f"from {[f'{k}:{len(v)}' for k, v in tb.items()]}")
        n = accumulate_grads(model, lossfn, dl, dev, amp_dt, a.batches, tag)
        res[tag] = rgn_table(model)
        counts[tag] = {"tiles": len(ds), "batches": n, "cities": cities}
        model.zero_grad(set_to_none=True)

    groups = sorted(set(res["ref"]) | set(res["shift"]), key=order_key)
    print(f"\n{'group':<14}{'params':>12}{'RGN ref':>12}{'RGN shift':>12}{'ratio':>9}")
    print("-" * 59)
    rows = []
    for g in groups:
        r, s = res["ref"].get(g), res["shift"].get(g)
        if not r or not s:
            continue
        ratio = s["rgn"] / r["rgn"] if r["rgn"] > 0 else float("nan")
        rows.append(dict(group=g, params=r["n"], rgn_ref=r["rgn"],
                         rgn_shift=s["rgn"], ratio=ratio))
        print(f"{g:<14}{r['n']:>12,}{r['rgn']:>12.3e}{s['rgn']:>12.3e}{ratio:>9.3f}")

    # The ratio's absolute LEVEL is not a shift measurement.  If the shifted domain
    # simply has a larger loss, every gradient scales up together and the whole column
    # lifts off 1.0 without any block being special -- exactly what the synthetic smoke
    # test shows (a uniform ~1.2-1.35 across all 12 blocks of a generator that has no
    # real domain shift in it).  Only the SHAPE across blocks carries information, so
    # divide it out and report a level-free column too.
    import statistics as _st
    med = _st.median([x["ratio"] for x in rows if x["ratio"] == x["ratio"]]) or 1.0
    for x in rows:
        x["ratio_norm"] = x["ratio"] / med
    print(f"\nmedian ratio across all groups = {med:.3f}  (divided out below; a column "
          f"flat at 1.00\nafter normalisation means the shift is NOT localised)")

    blocks = [x for x in rows if x["group"].startswith("blk")]
    if blocks:
        k = max(3, len(blocks) // 3)
        early = sum(x["ratio_norm"] for x in blocks[:k]) / k
        late = sum(x["ratio_norm"] for x in blocks[-k:]) / k
        mid = sum(x["ratio_norm"] for x in blocks[k:-k]) / max(1, len(blocks) - 2 * k)
        spread = max(x["ratio_norm"] for x in blocks) - min(x["ratio_norm"] for x in blocks)
        print(f"normalised  early(first {k}) {early:.3f}   "
              f"middle {mid:.3f}   late(last {k}) {late:.3f}   spread {spread:.3f}")
        if spread < 0.15:
            print("  -> spread is small.  No block stands out; surgical fine-tuning has\n"
                  "     no target here and the sweep should expect F3 (full) to win.")
        top = sorted(blocks, key=lambda x: -x["ratio_norm"])[:5]
        print("highest-ratio blocks: "
              + ", ".join(f"{x['group']}({x['ratio_norm']:.2f})" for x in top))
        print("\nRead it as: ratio > 1 means this group's gradient grew on the shifted\n"
              "domain relative to its own weights -- i.e. it is where the model is least\n"
              "adapted, and the surgical-fine-tuning candidate.  A flat column near 1.0\n"
              "means the shift is not localised and full fine-tuning is as good a choice\n"
              "as any.  This is ONE shift on ONE checkpoint: treat it as a prior for the\n"
              "block sweep, not as the answer.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(ckpt=a.ckpt, backbone=cfg.model.checkpoint, counts=counts,
                       splits=a.splits, rows=rows), f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
