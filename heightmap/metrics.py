"""Evaluation metrics.

The rubric names RMSE, MAE and correlation against a LiDAR reference.  Correlation is
invariant to scale and shift, so it isolates STRUCTURE -- multiply every prediction by
1.4 and add 3 m and r does not move while MAE and RMSE get much worse.  Since metric
anchoring is a separate stage downstream, r is the cleanest measure of this model's
own contribution.

Everything is reported twice: overall, and restricted to building pixels.  About a
third of pixels are ground at exactly 0 m and are nearly free, so an overall MAE
largely measures how well the model predicts zero.

bias is signed on purpose.  Two models can both score 4 m MAE: one randomly off by
+-4 m, the other consistently 4 m too short.  Only bias separates them, and the second
is the long-tail flattening problem, quantified.
"""
from __future__ import annotations
import numpy as np
import torch

from .data.classes import BUILDING

BANDS = [(0.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, float("inf"))]


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _core(pred, gt, m):
    if m.sum() < 2:
        d = {k: float("nan") for k in ("mae", "rmse", "bias", "r", "delta1")}
        d["n"] = int(m.sum())
        return d
    p, g = pred[m], gt[m]
    d = p - g
    out = {"mae": float(np.abs(d).mean()),
           "rmse": float(np.sqrt((d ** 2).mean())),
           "bias": float(d.mean()),
           "n": int(m.sum())}
    sp, sg = p.std(), g.std()
    out["r"] = float(np.corrcoef(p, g)[0, 1]) if sp > 1e-8 and sg > 1e-8 else float("nan")
    # delta1 on (h+1) so that ground (exactly 0 m) does not divide by zero.
    # Papers differ here; this choice is stated in the output so numbers stay comparable.
    a, b = p + 1.0, g + 1.0
    out["delta1"] = float((np.maximum(a / b, b / a) < 1.25).mean())
    return out


def evaluate(pred, gt, mask, sem=None, building_class: int = BUILDING) -> dict:
    """pred/gt (B,1,H,W) or (H,W); mask same; sem (B,H,W) class ids or None."""
    p, g = _np(pred).squeeze(), _np(gt).squeeze()
    m = _np(mask).squeeze().astype(bool)
    res = {"overall": _core(p, g, m), "delta1_convention": "on (h+1m) to avoid /0 at ground"}

    if sem is not None:
        s = _np(sem).squeeze()
        bm = m & (s == building_class)
        res["building"] = _core(p, g, bm)
    else:
        # fall back to a height-defined building proxy so the number always exists
        res["building"] = _core(p, g, m & (g > 2.0))
        res["building_note"] = "no semantic mask supplied; proxy = ground truth > 2 m"

    res["by_band"] = {}
    for lo, hi in BANDS:
        bm = m & (g >= lo) & (g < hi)
        key = f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}m"
        res["by_band"][key] = _core(p, g, bm)
    return res


def sharpness(pred, gt, mask, edge_pct: float = 98.0) -> dict:
    """How sharp are predicted edges, at the places the truth actually has edges?

    Note the subtlety: MEAN gradient magnitude does NOT measure sharpness, because total
    variation is preserved when a step is blurred into a ramp -- the same total spread
    over more pixels.  What blurring does is move gradient AWAY from the true edge.  So
    the measurement has to be conditioned on where the truth's edges are.

    ratio < 1 means the prediction is smoother than the truth at its edges, which is the
    'melted building' failure the multi-scale gradient loss exists to prevent.
    """
    p, g = _np(pred).squeeze(), _np(gt).squeeze()
    m = _np(mask).squeeze().astype(bool)

    def grad(a):
        dx = np.abs(np.diff(a, axis=-1))[..., :-1, :]
        dy = np.abs(np.diff(a, axis=-2))[..., :, :-1]
        return dx + dy

    gp, gg = grad(p), grad(g)
    mm = m[..., :-1, :-1]
    if mm.sum() < 10:
        return {"grad_pred_at_edges": float("nan"), "grad_gt_at_edges": float("nan"),
                "ratio": float("nan"), "n_edge": 0}
    thr = np.percentile(gg[mm], edge_pct)
    edge = mm & (gg >= max(thr, 1e-6))
    if edge.sum() < 10:
        return {"grad_pred_at_edges": float("nan"), "grad_gt_at_edges": float("nan"),
                "ratio": float("nan"), "n_edge": int(edge.sum())}
    a, b = float(gp[edge].mean()), float(gg[edge].mean())
    return {"grad_pred_at_edges": a, "grad_gt_at_edges": b,
            "ratio": a / b if b > 1e-9 else float("nan"), "n_edge": int(edge.sum())}


def format_table(res: dict) -> str:
    L = []
    hdr = f"{'scope':<14}{'MAE':>8}{'RMSE':>9}{'bias':>9}{'r':>8}{'d1':>8}{'n':>12}"
    L.append(hdr); L.append("-" * len(hdr))
    for scope in ("overall", "building"):
        c = res[scope]
        L.append(f"{scope:<14}{c['mae']:>8.3f}{c['rmse']:>9.3f}{c['bias']:>+9.3f}"
                 f"{c['r']:>8.3f}{c['delta1']:>8.3f}{c['n']:>12,}")
    L.append("")
    L.append(f"{'band':<14}{'MAE':>8}{'RMSE':>9}{'bias':>9}{'r':>8}{'d1':>8}{'n':>12}")
    for k, c in res["by_band"].items():
        L.append(f"{k:<14}{c['mae']:>8.3f}{c['rmse']:>9.3f}{c['bias']:>+9.3f}"
                 f"{c['r']:>8.3f}{c['delta1']:>8.3f}{c['n']:>12,}")
    return "\n".join(L)


class MetricAccumulator:
    """Exact streaming metrics over a whole validation set.

    Pearson r cannot be averaged across batches, so the sufficient statistics are
    accumulated instead: n, sum(p), sum(g), sum(p^2), sum(g^2), sum(pg) reconstruct it
    exactly.  The alternative -- averaging per-batch r -- is simply wrong.
    """

    SCOPES = ("overall", "building") + tuple(
        f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}m" for lo, hi in BANDS)

    def __init__(self, building_class: int = BUILDING):
        self.building_class = building_class
        self.s = {k: dict(n=0, sd=0.0, sad=0.0, sd2=0.0, sp=0.0, sg=0.0,
                          sp2=0.0, sg2=0.0, spg=0.0, d1=0) for k in self.SCOPES}

    def _add(self, key, p, g):
        if p.size == 0:
            return
        a = self.s[key]
        d = p - g
        a["n"] += p.size
        a["sd"] += float(d.sum()); a["sad"] += float(np.abs(d).sum()); a["sd2"] += float((d ** 2).sum())
        a["sp"] += float(p.sum()); a["sg"] += float(g.sum())
        a["sp2"] += float((p ** 2).sum()); a["sg2"] += float((g ** 2).sum())
        a["spg"] += float((p * g).sum())
        x, y = p + 1.0, g + 1.0
        a["d1"] += int((np.maximum(x / y, y / x) < 1.25).sum())

    def update(self, pred, gt, mask, sem=None):
        p, g = _np(pred).squeeze(), _np(gt).squeeze()
        m = _np(mask).squeeze().astype(bool)
        self._add("overall", p[m], g[m])
        if sem is not None:
            s = _np(sem).squeeze()
            bm = m & (s == self.building_class)
        else:
            bm = m & (g > 2.0)
        self._add("building", p[bm], g[bm])
        for lo, hi in BANDS:
            key = f"{lo:g}-{'inf' if hi == float('inf') else f'{hi:g}'}m"
            k = m & (g >= lo) & (g < hi)
            self._add(key, p[k], g[k])

    def _finish(self, a):
        n = a["n"]
        if n < 2:
            d = {k: float("nan") for k in ("mae", "rmse", "bias", "r", "delta1")}
            d["n"] = n
            return d
        cov = a["spg"] / n - (a["sp"] / n) * (a["sg"] / n)
        vp = max(a["sp2"] / n - (a["sp"] / n) ** 2, 0.0)
        vg = max(a["sg2"] / n - (a["sg"] / n) ** 2, 0.0)
        r = cov / np.sqrt(vp * vg) if vp > 1e-12 and vg > 1e-12 else float("nan")
        return {"mae": a["sad"] / n, "rmse": float(np.sqrt(a["sd2"] / n)),
                "bias": a["sd"] / n, "r": float(r), "delta1": a["d1"] / n, "n": n}

    def result(self) -> dict:
        out = {k: self._finish(v) for k, v in self.s.items()}
        res = {"overall": out["overall"], "building": out["building"],
               "by_band": {k: out[k] for k in self.SCOPES[2:]},
               "delta1_convention": "on (h+1m) to avoid /0 at ground"}
        return res
