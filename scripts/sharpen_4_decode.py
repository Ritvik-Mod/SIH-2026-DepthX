#!/usr/bin/env python
"""IDEA 6 -- decode the bin distribution differently.  No retraining.

The whole argument for BinsHeightHead is in its own docstring: at a roof edge the truth
is bimodal -- 0 m or 18 m, never the average -- and direct regression is forced to emit
that average, a value in mid-air, which is what ramps the edge.  A distribution over
bins can stay bimodal.

But heads.py:110 then does

    h = (p * centres).sum(dim=1)

which is the EXPECTATION of that distribution.  The expectation of a bimodal
distribution is also the value in mid-air.  The head represents the bimodality
correctly and throws it away on the last line.

This script reads the same weights and decodes p differently:

  mean     the shipped rule, E[h].  The baseline.
  mode     centres[argmax p].  Commits to one side of the edge.  Costs sub-bin
           precision everywhere else -- quantisation to the bin grid.
  temp:T   softmax(logits / T).  T<1 sharpens continuously between the two; the
           limit T->0 is mode.  Keeps sub-bin precision where p is already unimodal.
  win:k    expectation restricted to the k bins either side of the argmax,
           renormalised.  Discards the far mode but keeps sub-bin interpolation
           within the near one -- the middle ground, and the one to beat.

Two controls that decide whether any of this can work, printed before the table:

  * mean+bilinear MUST reproduce out["height"] to float precision.  bin_logits come
    out at decoder resolution while height is collapsed BEFORE the upsample, so this
    script re-implements the tail of the head; if the control fails, nothing below
    means anything.
  * if p is already unimodal at true edges, mode == mean and the hypothesis is dead
    on arrival.  |mode - mean| at edge pixels is the measurement that says so.

Upsampling is swept too, because it is not neutral: a mode-decoded field that is then
bilinearly resampled 296 -> 518 has had a ramp put back into every edge it just made
sharp.  Reading a sharpness number without that column is how you fool yourself.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F

from _sharpen_common import build, evaluate, row_str, header
from heightmap.data.transforms import denormalise  # noqa: F401  (parity with forward_batch)


# ---- decode rules --------------------------------------------------------------
def decode(logits: torch.Tensor, centres: torch.Tensor, rule: str) -> torch.Tensor:
    """logits (N,h,w), centres (N,) -> height (h,w) at DECODER resolution."""
    c = centres[:, None, None]
    if rule == "mean":
        return (torch.softmax(logits, 0) * c).sum(0)
    if rule == "mode":
        return centres[logits.argmax(0)]
    if rule.startswith("temp:"):
        T = float(rule.split(":", 1)[1])
        return (torch.softmax(logits / T, 0) * c).sum(0)
    if rule.startswith("win:"):
        k = int(rule.split(":", 1)[1])
        idx = logits.argmax(0)                                    # (h,w)
        bins = torch.arange(logits.shape[0], device=logits.device)
        keep = (bins[:, None, None] - idx[None]).abs() <= k       # (N,h,w)
        p = torch.softmax(logits, 0) * keep
        p = p / p.sum(0, keepdim=True).clamp(min=1e-12)
        return (p * c).sum(0)
    raise ValueError(f"unknown decode rule '{rule}'")


def upsample(h: torch.Tensor, out_hw, mode: str) -> np.ndarray:
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(h[None, None], out_hw, mode=mode, **kw)[0, 0].numpy()


# ---- diagnostics ---------------------------------------------------------------
def bimodality(logits, centres, gt, edge_pct=98.0):
    """Is p actually bimodal where the truth has edges?  If not, stop here.

    Edge locations come from the TRUTH, area-downsampled to decoder resolution -- the
    same conditioning heightmap.metrics.sharpness uses, since gradient conditioned on
    the prediction would just measure where the prediction already decided to be sharp.
    """
    N, h, w = logits.shape
    gd = F.interpolate(torch.from_numpy(gt)[None, None], (h, w), mode="area")[0, 0]
    g = (gd.diff(dim=-1)[:-1, :].abs() + gd.diff(dim=-2)[:, :-1].abs())
    thr = np.percentile(g.numpy(), edge_pct)
    edge = torch.zeros(h, w, dtype=torch.bool)
    edge[:-1, :-1] = g >= max(thr, 1e-6)

    p = torch.softmax(logits, 0)
    mean = (p * centres[:, None, None]).sum(0)
    mode = centres[logits.argmax(0)]
    gap = (mode - mean).abs()
    ent = -(p.clamp_min(1e-12).log() * p).sum(0)              # nats
    # the quantisation step a mode decode would actually incur: the width of the bin
    # each pixel lands in, not the average width over a 250 m range nobody occupies.
    widths = torch.empty_like(centres)
    widths[1:-1] = (centres[2:] - centres[:-2]) / 2
    widths[0] = centres[1] - centres[0]
    widths[-1] = centres[-1] - centres[-2]
    wpix = widths[logits.argmax(0)]

    def m(t, sel=None):
        return float(t[sel].mean()) if sel is None or sel.any() else float("nan")

    return dict(gap_all=m(gap), gap_edge=m(gap, edge),
                gap1_all=float((gap > 1.0).float().mean()),
                gap1_edge=float((gap[edge] > 1.0).float().mean()) if edge.any() else np.nan,
                eff_bins_all=float(ent.exp().mean()), eff_bins_edge=m(ent.exp(), edge),
                width_pix=m(wpix), width_mean=float(widths.mean()), n_edge=int(edge.sum()))


@torch.no_grad()
def forward_bins(model, ds, i, dev, use_gsd=True):
    b = ds[i]
    out = model(b["image"][None].to(dev), b["gsd"][None].to(dev) if use_gsd else None)
    if "bin_logits" not in out:
        sys.exit("this checkpoint has a regression head; the test needs model.head=bins")
    return dict(logits=out["bin_logits"][0].float().cpu(),
                centres=out["bin_centres"][0].float().cpu(),
                shipped=out["height"][0, 0].float().cpu().numpy(),
                gt=b["height"][0].numpy(), mask=b["mask"][0].numpy().astype(bool),
                cls=b["cls"].numpy(), tid=b["tile_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--rules", nargs="*",
                    default=["mean", "temp:0.5", "temp:0.25", "temp:0.1",
                             "win:1", "win:2", "win:4", "mode"])
    ap.add_argument("--up", nargs="*", default=["bilinear", "nearest"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    model, cfg, ds, dev = build(a.ckpt, a.root, a.split, a.n, a.device)
    recs = [forward_bins(model, ds, i, dev, use_gsd=bool(cfg.model.use_gsd))
            for i in range(len(ds))]
    N, dh, dw = recs[0]["logits"].shape
    out_hw = recs[0]["gt"].shape
    print(f"{len(recs)} tiles from {a.split}   {N} bins   decoder {dh}x{dw} -> "
          f"output {out_hw[0]}x{out_hw[1]}\n")

    # ---- control 1: does mean+bilinear reproduce the shipped field? -------------
    err = max(float(np.abs(upsample(decode(r["logits"], r["centres"], "mean"),
                                    out_hw, "bilinear") - r["shipped"]).max())
              for r in recs)
    ok = err < 1e-3
    print(f"CONTROL  mean+bilinear vs out['height']: max |diff| = {err:.2e} m  "
          f"{'OK' if ok else 'MISMATCH -- everything below is meaningless'}")
    if not ok:
        sys.exit(1)

    # ---- control 2: is p bimodal at true edges at all? -------------------------
    d = [bimodality(r["logits"], r["centres"], r["gt"]) for r in recs]
    agg = {k: float(np.nanmean([x[k] for x in d])) for k in d[0]}
    print(f"CONTROL  |mode - mean|:  all pixels {agg['gap_all']:.3f} m   "
          f"at true edges {agg['gap_edge']:.3f} m")
    print(f"         fraction > 1 m: all {agg['gap1_all']:.3f}   "
          f"at true edges {agg['gap1_edge']:.3f}")
    print(f"         effective bins exp(H): all {agg['eff_bins_all']:.2f}   "
          f"at true edges {agg['eff_bins_edge']:.2f}   (1.0 = already committed)")
    print(f"         bin width at the argmax {agg['width_pix']:.3f} m  "
          f"(mean over all {N} bins {agg['width_mean']:.3f} m)")
    print(f"         a mode decode quantises to that grid: RMSE floor ~"
          f"{agg['width_pix'] / np.sqrt(12):.3f} m\n")

    gts = [r["gt"] for r in recs]
    masks = [r["mask"] for r in recs]
    clss = [r["cls"] for r in recs]
    print(header())
    for up in a.up:
        for rule in a.rules:
            preds = [upsample(decode(r["logits"], r["centres"], rule), out_hw, up)
                     for r in recs]
            print(row_str(f"{rule} + {up}", evaluate(preds, gts, masks, clss)))
        print("-" * 118)

    print("\n'mean + bilinear' is the shipped model; every other row is the SAME weights.\n"
          "A rule wins only if sharp rises while building MAE and 30m+ bias do not.\n"
          "If |mode - mean| at true edges is small, p is unimodal there, the head never\n"
          "learned the bimodality the bins exist to represent, and no decode rule can\n"
          "recover it -- that is a training-objective finding, not a decoding one.")


if __name__ == "__main__":
    main()
