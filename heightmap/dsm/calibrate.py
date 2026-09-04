"""Scale calibration -- PS requirement #3.

The height model already outputs metres, so this is not "invent a scale from scratch".
It is residual correction, and there is measured evidence it is needed: on the held-out
New York test the model kept building heights almost perfectly (1.585 m MAE, r = 0.901)
while the whole scene sat 2.02 m too low.  A near-perfect shape with a constant offset
is exactly the error a linear fit removes.

Fit on GROUND, not on buildings.  The building heights are the part that is already
right; fitting there spends the fit on the good half and leaves the offset in place.
Ground is class 1 and road is class 5 in the GAMUS labels used throughout this repo.
"""
from __future__ import annotations

import numpy as np

from ..data.classes import GROUND, ROAD

#: Land-cover classes that are, by definition, at ~0 m above local ground.
GROUND_CLASSES = (GROUND, ROAD)

#: Below this many reference samples a free scale factor fits noise rather than signal,
#: so calibrate() drops to offset-only and says so in the returned report.
MIN_SAMPLES_FOR_SCALE = 50


def calibrate(agl, reference_values, reference_mask, fit_scale: bool = True):
    """Least-squares fit of  agl_calibrated = a * agl + b  against a reference.

    The reference can be GCP elevations, a DEM over open ground, or LiDAR where it
    exists -- anything you trust more than the model.

    fit_scale=False fits the offset only.  Prefer that when you have few reference
    points: with a handful of samples a free scale factor will happily fit noise and
    make things worse.  This function enforces that automatically below
    MIN_SAMPLES_FOR_SCALE and records the downgrade in the report.

    Returns (calibrated_agl, report).
    """
    agl = np.asarray(agl, np.float64)
    reference_values = np.asarray(reference_values, np.float64)
    mask = np.asarray(reference_mask, bool)

    p = agl[mask].ravel()
    g = reference_values[mask].ravel()
    good = np.isfinite(p) & np.isfinite(g)
    p, g = p[good], g[good]

    if p.size == 0:
        return (np.asarray(agl, np.float32),
                {"scale": 1.0, "offset": 0.0, "n_reference_points": 0,
                 "applied": False,
                 "note": "no usable reference samples; left uncalibrated"})

    downgraded = False
    if fit_scale and p.size < MIN_SAMPLES_FOR_SCALE:
        fit_scale, downgraded = False, True

    if fit_scale:
        A = np.column_stack([p, np.ones_like(p)])
        (a, b), *_ = np.linalg.lstsq(A, g, rcond=None)
    else:
        a, b = 1.0, float(np.mean(g - p))

    resid = g - (a * p + b)
    report = {
        "scale": float(a),
        "offset": float(b),
        "n_reference_points": int(p.size),
        "applied": True,
        "fit": "scale+offset" if fit_scale else "offset only",
        "residual_mae_m": round(float(np.mean(np.abs(resid))), 4),
        "residual_rmse_m": round(float(np.sqrt(np.mean(resid ** 2))), 4),
        "bias_before_m": round(float(np.mean(p - g)), 4),
    }
    if downgraded:
        report["note"] = (f"only {p.size} reference points (< {MIN_SAMPLES_FOR_SCALE}); "
                          "fitted the offset only, because a free scale factor on this "
                          "few samples fits noise")
    return (a * agl + b).astype(np.float32), report


def ground_mask(semantic, classes=GROUND_CLASSES):
    """Pixels that should read ~0 m AGL: ground and road."""
    semantic = np.asarray(semantic)
    m = np.zeros(semantic.shape, bool)
    for c in classes:
        m |= (semantic == c)
    return m


def calibrate_to_flat_ground(agl, semantic, valid=None, fit_scale: bool = False):
    """Self-calibration with no external data at all.

    Ground and road pixels are ~0 m above local ground by definition, so the model's
    own semantic map is a free reference: whatever the model predicts there is, on
    average, its bias.  Subtracting it removes exactly the kind of whole-scene offset
    measured on New York.

    fit_scale defaults to False here on purpose.  The reference is a constant 0, so a
    free scale factor has nothing to lean on -- it would simply crush the heights
    towards zero.  Only the offset is identifiable from this reference.

    Returns (calibrated_agl, report).
    """
    agl = np.asarray(agl, np.float64)
    m = ground_mask(semantic)
    if valid is not None:
        m &= np.asarray(valid, bool)
    zeros = np.zeros_like(agl)
    out, report = calibrate(agl, zeros, m, fit_scale=fit_scale)
    report["reference"] = "self: ground+road pixels assumed 0 m AGL"
    report["ground_pixel_fraction"] = round(float(m.mean()), 4)
    return out, report


def metrics(pred, truth, valid=None, semantic=None):
    """RMSE / MAE / correlation / signed bias -- PS requirement #13.

    Reported per land-cover group as well as pooled, because roughly a third of pixels
    are ground sitting at almost exactly 0 m and are nearly free to get right.  A
    pooled number flatters you.

    Signed bias is reported alongside MAE deliberately.  Two models can both score 4 m
    MAE: one randomly off by +/-4 m, the other consistently 4 m short.  Only the sign
    separates them, and a consistent offset is the fixable kind.
    """
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    base = np.isfinite(pred) & np.isfinite(truth)
    if valid is not None:
        base &= np.asarray(valid, bool)

    def _one(m):
        if m.sum() < 2:
            return {"n": int(m.sum())}
        p, g = pred[m], truth[m]
        d = p - g
        sp, sg = p.std(), g.std()
        r = float(np.corrcoef(p, g)[0, 1]) if sp > 1e-9 and sg > 1e-9 else None
        return {
            "n": int(m.sum()),
            "rmse_m": round(float(np.sqrt(np.mean(d ** 2))), 4),
            "mae_m": round(float(np.mean(np.abs(d))), 4),
            "bias_m": round(float(np.mean(d)), 4),      # signed: + = we predict too high
            "correlation": round(r, 4) if r is not None else None,
        }

    out = {"all": _one(base)}
    if semantic is not None:
        semantic = np.asarray(semantic)
        from ..data.classes import BUILDING, NAMES
        out["ground"] = _one(base & ground_mask(semantic))
        out["building"] = _one(base & (semantic == BUILDING))
        out["per_class"] = {NAMES.get(int(c), str(int(c))): _one(base & (semantic == c))
                            for c in np.unique(semantic[base]) if int(c) != 0}
    return out
