"""
dw_tda.metrics
==============
Baseline-vs-refined measurement, including the topology-aware side.

Height metrics are computed on valid pixels only.  "Building" metrics use a
building mask; if no semantic mask is supplied, one is derived from the ground
truth by thresholding height, which is documented rather than hidden because it
is not identical to the GAMUS semantic mask and will differ by a few percent.

Sharpness follows the pipeline's own convention: the ratio of mean gradient
magnitude on the strong-edge pixels of the truth.  1.0 means the prediction is
as sharp as the truth; the shipped model measures 0.084-0.133.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from . import tda_core as T


def valid_mask(*arrays: np.ndarray, nodata: float = T.NODATA) -> np.ndarray:
    m = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        m &= np.isfinite(a) & ~np.isclose(a, nodata)
    return m


def height_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() < 10:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "bias": float("nan"), "r": float("nan")}
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    d = p - g
    r = float(np.corrcoef(p, g)[0, 1]) if p.std() > 0 and g.std() > 0 else float("nan")
    return {
        "n": int(mask.sum()),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "bias": float(d.mean()),
        "r": r,
    }


def sharpness_ratio(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                    pct: float = 98.0) -> float:
    gy, gx = np.gradient(gt)
    gg = np.hypot(gx, gy)
    py, px = np.gradient(pred)
    pg = np.hypot(px, py)
    sel = mask & (gg > np.percentile(gg[mask], pct))
    if sel.sum() < 20:
        return float("nan")
    return float(pg[sel].mean() / (gg[sel].mean() + 1e-9))


def gradient_stats(x: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    gy, gx = np.gradient(x)
    gm = np.hypot(gx, gy)[mask]
    return {
        "grad_mean": float(gm.mean()),
        "grad_p99": float(np.percentile(gm, 99)),
        "tv": float(gm.sum()),
    }


def edge_f1(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
            tol_px: int = 2, pct: float = 98.0) -> Dict[str, float]:
    """Precision/recall of strong height edges within `tol_px`.  Catches the
    failure mode where MAE improves while edges move to the wrong place."""
    def edges(x):
        gy, gx = np.gradient(x)
        gm = np.hypot(gx, gy)
        thr = np.percentile(gm[mask], pct)
        return (gm > thr) & mask
    e_p, e_g = edges(pred), edges(gt)
    if e_g.sum() == 0 or e_p.sum() == 0:
        return {"edge_precision": float("nan"), "edge_recall": float("nan"),
                "edge_f1": float("nan")}
    dg = ndi.distance_transform_edt(~e_g)
    dp = ndi.distance_transform_edt(~e_p)
    prec = float((dg[e_p] <= tol_px).mean())
    rec = float((dp[e_g] <= tol_px).mean())
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1}


def edge_localisation(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray,
                      thr_m_per_px: float = 0.6, tol_px: int = 2) -> Dict[str, float]:
    """Edge agreement at a FIXED absolute gradient threshold.

    Preferred over the percentile version: a percentile threshold adapts to the
    field it is measuring, so sharpening a blurred prediction changes which
    pixels are selected and the score moves for reasons that have nothing to do
    with edge placement.  On the synthetic scenes the percentile F1 fell 0.926
    -> 0.888 while the absolute F1 rose 0.950 -> 0.981 and the mean distance
    from a predicted edge to the nearest true edge halved.  Report both.
    """
    def E(x):
        gy, gx = np.gradient(x)
        return (np.hypot(gx, gy) > thr_m_per_px) & mask
    ep, eg = E(pred), E(gt)
    if ep.sum() < 10 or eg.sum() < 10:
        return {"edge_abs_f1": float("nan"), "edge_dist_px": float("nan")}
    dg = ndi.distance_transform_edt(~eg)
    dp = ndi.distance_transform_edt(~ep)
    pr = float((dg[ep] <= tol_px).mean())
    rc = float((dp[eg] <= tol_px).mean())
    return {
        "edge_abs_precision": pr,
        "edge_abs_recall": rc,
        "edge_abs_f1": 2 * pr * rc / (pr + rc + 1e-9),
        "edge_dist_px": float(dg[ep].mean()),
    }


# ---------------------------------------------------------------------------
# topology metrics
# ---------------------------------------------------------------------------


def wasserstein1(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
    """1-Wasserstein distance between 0-dim persistence diagrams under the
    L-infinity ground metric, with the usual diagonal augmentation (an unmatched
    point is transported to the diagonal at half its persistence).  Exact via
    Hungarian assignment; fine for a few hundred points per diagram."""
    a = np.atleast_2d(dgm_a).astype(np.float64)
    b = np.atleast_2d(dgm_b).astype(np.float64)
    a = a[np.isfinite(a).all(1)] if a.size else a.reshape(0, 2)
    b = b[np.isfinite(b).all(1)] if b.size else b.reshape(0, 2)
    na, nb = len(a), len(b)
    if na == 0 and nb == 0:
        return 0.0
    n = na + nb
    C = np.zeros((n, n))
    if na and nb:
        C[:na, :nb] = np.max(np.abs(a[:, None, :] - b[None, :, :]), axis=2)
    da = np.abs(a[:, 0] - a[:, 1]) / 2 if na else np.zeros(0)
    db = np.abs(b[:, 0] - b[:, 1]) / 2 if nb else np.zeros(0)
    if na:
        C[:na, nb:] = np.inf
        C[np.arange(na), nb + np.arange(na)] = da
    if nb:
        C[na:, :nb] = np.inf
        C[na + np.arange(nb), np.arange(nb)] = db
    C[na:, nb:] = 0.0
    C[~np.isfinite(C)] = 1e9
    ri, ci = linear_sum_assignment(C)
    return float(C[ri, ci].sum())


def topology_metrics(pred: np.ndarray, gt: np.ndarray,
                     min_persistence: float = 0.5,
                     max_points: int = 400) -> Dict[str, float]:
    dp = T.persistence_diagram(pred, min_persistence)
    dg = T.persistence_diagram(gt, min_persistence)

    def top(d):
        if len(d) <= max_points:
            return d
        k = np.argsort(-(d[:, 0] - d[:, 1]))[:max_points]
        return d[k]

    dp_t, dg_t = top(dp), top(dg)
    lv = np.linspace(min(pred.min(), gt.min()), max(np.percentile(pred, 99.9),
                                                    np.percentile(gt, 99.9)), 24)
    bp = T.betti_curve(pred, lv)
    bg = T.betti_curve(gt, lv)
    return {
        "n_features_pred": int(len(dp)),
        "n_features_gt": int(len(dg)),
        "n_feature_error": int(abs(len(dp) - len(dg))),
        "wasserstein1": wasserstein1(dp_t, dg_t),
        "betti_l1": float(np.abs(bp - bg).mean()),
        "total_persistence_pred": float((dp[:, 0] - dp[:, 1]).sum()) if len(dp) else 0.0,
        "total_persistence_gt": float((dg[:, 0] - dg[:, 1]).sum()) if len(dg) else 0.0,
    }


def evaluate(pred: np.ndarray, gt: np.ndarray,
             building_mask: Optional[np.ndarray] = None,
             building_height_m: float = 2.0,
             with_topology: bool = True) -> Dict[str, float]:
    m = valid_mask(pred, gt)
    if building_mask is None:
        building_mask = (gt >= building_height_m) & m
        derived = True
    else:
        building_mask = building_mask & m
        derived = False
    out: Dict[str, float] = {"building_mask_derived_from_gt_height": derived}
    for k, v in height_metrics(pred, gt, m).items():
        out[f"overall_{k}"] = v
    for k, v in height_metrics(pred, gt, building_mask).items():
        out[f"building_{k}"] = v
    out["sharpness_ratio"] = sharpness_ratio(pred, gt, m)
    out.update(edge_f1(pred, gt, m))
    out.update(edge_localisation(pred, gt, m))
    gp = gradient_stats(pred, m)
    gg = gradient_stats(gt, m)
    out["grad_mean_pred"] = gp["grad_mean"]
    out["grad_mean_gt"] = gg["grad_mean"]
    out["tv_ratio"] = gp["tv"] / (gg["tv"] + 1e-9)
    if with_topology:
        out.update(topology_metrics(pred, gt))
    return out
