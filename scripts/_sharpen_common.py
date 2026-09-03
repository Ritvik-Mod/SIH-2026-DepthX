"""Shared harness for the three sharpening experiments.

Evaluation runs on 518 centre crops through GamusDataset(train=False), exactly as
heightmap.train.validate does, so every number here is directly comparable to the rows
in the ablation table rather than approximately comparable.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch

from heightmap import config
from heightmap.data.gamus import GamusDataset, index_tiles
from heightmap.data.classes import BUILDING
from heightmap.metrics import sharpness
from heightmap.predict import load_model
from heightmap.utils.misc import pick_device

BANDS = [(0.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, np.inf)]


def build(ckpt, root, split, n, device="auto", tiles=None):
    """-> (model, cfg, dataset, device).  Loads the checkpoint's own config."""
    dev = pick_device() if device == "auto" else torch.device(device)
    model, cfg = load_model(ckpt, dev)
    cfg = config.OmegaConf.merge(cfg, {"data": {"root": root, "sun_cache": ""}}) \
        if hasattr(config, "OmegaConf") else cfg
    cfg.data.root = root
    ids = tiles or [t for _c, t in index_tiles(root, split)][:n]
    ds = GamusDataset(cfg, split, train=False, tiles=[(t.split("_")[0], t) for t in ids])
    return model, cfg, ds, dev


@torch.no_grad()
def forward_batch(model, ds, i, dev, use_gsd=True):
    """-> dict of numpy arrays for one tile: pred, gt, mask, cls, sem, rgb01."""
    b = ds[i]
    x = b["image"][None].to(dev)
    gsd = b["gsd"][None].to(dev)
    out = model(x, gsd if use_gsd else None)
    r = dict(pred=out["height"][0, 0].float().cpu().numpy(),
             gt=b["height"][0].numpy(), mask=b["mask"][0].numpy().astype(bool),
             cls=b["cls"].numpy(), tid=b["tile_id"])
    if "semantic" in out:
        r["sem_logits"] = out["semantic"][0].float().cpu().numpy()
        r["sem"] = r["sem_logits"].argmax(0)
    from heightmap.data.transforms import denormalise
    r["rgb01"] = denormalise(b["image"][None])[0].permute(1, 2, 0).numpy()
    return r


def _stat(p, g, m):
    if m.sum() < 16:
        return dict(n=0, mae=np.nan, bias=np.nan, r=np.nan)
    a, b = p[m], g[m]
    rr = np.nan
    if a.std() > 1e-6 and b.std() > 1e-6:
        rr = float(np.corrcoef(a, b)[0, 1])
    return dict(n=int(m.sum()), mae=float(np.abs(a - b).mean()),
                bias=float((a - b).mean()), r=rr)


def evaluate(preds, gts, masks, clss):
    """Accumulate over tiles.  Returns the row printed by row_str()."""
    P = np.concatenate([p[m] for p, m in zip(preds, masks)])
    G = np.concatenate([g[m] for g, m in zip(gts, masks)])
    B = np.concatenate([(c == BUILDING)[m] for c, m in zip(clss, masks)])
    out = {"overall": _stat(P, G, np.ones_like(B, bool)), "building": _stat(P, G, B)}
    for lo, hi in BANDS:
        sel = (G >= lo) & (G < hi)
        out[f"{lo:g}-{hi:g}m"] = _stat(P, G, sel)
    sh = [sharpness(p, g, m)["ratio"] for p, g, m in zip(preds, gts, masks)]
    out["sharpness"] = float(np.nanmean(sh))
    return out


def row_str(tag, r):
    o, b, t = r["overall"], r["building"], r["30-infm"]
    return (f"{tag:<26} MAE {o['mae']:6.3f}  r {o['r']:5.3f} | "
            f"bMAE {b['mae']:6.3f}  bBias {b['bias']:+6.3f}  b_r {b['r']:5.3f} | "
            f"30m+ MAE {t['mae']:6.3f} bias {t['bias']:+7.3f} | "
            f"sharp {r['sharpness']:5.3f}")


def header():
    return ("rows are the SAME statistics as the ablation table.  'sharp' is gradient at\n"
            "TRUE-edge locations / truth gradient there; 1.0 would match the truth.\n"
            "A gain in sharp that costs MAE or 30m+ bias is a loss, not a win.\n" + "-" * 118)
