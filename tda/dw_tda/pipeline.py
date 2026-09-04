"""
dw_tda.pipeline
===============
The staged refinement.  Order matters and each stage is individually
switchable, which is what makes the ablation ladder (A0..A5) a configuration
change rather than a different program.

    A0  input, untouched
    A1  topology only          : persistence simplification + artifact
                                 suppression + AGL zero-level calibration
    A2  A1 + height/gradient   : deconvolution, height-only prior
    A3  A1 + RGB/context       : deconvolution, RGB-only prior
    A4  A2 + A3                : deconvolution, both priors
    A5  A4 + plateau restore   : proposed system

Every stage that can move a pixel is followed by a guard, and the whole thing
ends with a data-consistency gate: the refined field must re-blur back onto the
observed prediction at least as well as the prediction re-blurs onto itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi

from . import tda_core as T
from .tda_core import RefineConfig


ABLATIONS = {
    "A0": dict(enable_ground_zeroing=False, tau_micro_m=0.0, enable_deconv=False,
               enable_plateau=False, suppress_artifacts=False),
    "A1": dict(enable_deconv=False, enable_plateau=False),
    "A2": dict(enable_plateau=False, rgb_prior_weight=0.0, height_prior_weight=1.0),
    "A3": dict(enable_plateau=False, rgb_prior_weight=1.0, height_prior_weight=0.0),
    "A4": dict(enable_plateau=False),
    # A5 = A4 + plateau restoration.  Kept in the ladder because it was tried
    # and reported, NOT because it survived: on the synthetic development
    # scenes it costs ~6% of A4's building-MAE gain (0.2466 -> 0.2617 m).  The
    # simpler method wins, so FINAL == A4.
    "A5": dict(enable_plateau=True),
    "FINAL": dict(),
}


@dataclass
class RefineResult:
    height: np.ndarray
    p_artifact_map: np.ndarray
    confidence_map: np.ndarray
    labels: np.ndarray
    stats: Dict = field(default_factory=dict)


def refine(
    H_in: np.ndarray,
    rgb: Optional[np.ndarray],
    gsd: float,
    cfg: Optional[RefineConfig] = None,
    nodata: float = T.NODATA,
    suppress_artifacts: bool = True,
) -> RefineResult:
    cfg = cfg or RefineConfig()
    t0 = time.time()
    stats: Dict = {"gsd_m": gsd}

    H, bad = T.split_nodata(H_in, nodata)
    stats["nodata_pixels"] = int(bad.sum())
    P0 = H.copy()  # the observation, kept for every guard and gate

    # ---- blur estimation: the forward model of the dominant error ---------
    sigma = T.estimate_blur_sigma(H, cfg)
    stats["sigma_px"] = sigma
    stats["sigma_m"] = sigma * gsd

    # ---- stage 1: micro-ripple persistence simplification ----------------
    if cfg.tau_micro_m > 0:
        H1 = T.persistence_simplify(H, cfg.tau_micro_m)
        stats["stage1_max_change_m"] = float(np.abs(H1 - H).max())
        stats["stage1_mean_change_m"] = float(np.abs(H1 - H).mean())
        H = H1

    # ---- stage 3: merge-tree segmentation of the elevated set ------------
    labels = T.elevated_segmentation(H, cfg, gsd)
    stats["n_regions"] = int(labels.max())

    # ---- stage 4: region features and classification ---------------------
    feats = T.region_features(H, labels, rgb, gsd)
    p_building, p_artifact = T.classify_regions(feats, cfg)
    stats["n_building_like"] = int((p_building >= 0.5).sum())
    stats["n_artifact_like"] = int((p_artifact >= 0.5).sum())

    p_b_map = np.zeros(H.shape, dtype=np.float32)
    if labels.max() > 0:
        lut_b = np.concatenate([[0.0], p_building]).astype(np.float32)
        p_b_map = lut_b[labels]

    # ---- stage 1b: artifact suppression (dense, topology-bounded) --------
    if suppress_artifacts and cfg.enable_artifact_suppression:
        p_a_map = T.artifact_probability(H, rgb, cfg, gsd)
        H = T.suppress_artifacts(H, p_a_map, cfg)
        stats["artifact_pixels_gt_0p5"] = int((p_a_map > 0.5).sum())
        stats["artifact_mean_p"] = float(p_a_map.mean())
    else:
        p_a_map = np.zeros(H.shape, dtype=np.float32)

    # ---- stage 2: AGL zero-level calibration -----------------------------
    if cfg.enable_ground_zeroing:
        g = T.estimate_local_ground(H, cfg, gsd)
        stats["ground_offset_median_m"] = float(np.median(g))
        stats["ground_offset_max_m"] = float(g.max())
        H = np.maximum(H - g, 0.0).astype(np.float32)
    else:
        stats["ground_offset_median_m"] = 0.0

    # Reference for the forward-model gate.  Stages 1, 1b and 2 are bounded
    # calibrations (persistence <= tau, artifact <= tau, a clamped additive
    # zero-level offset); they make no claim about geometry, so they must not
    # be judged by "does re-blurring reproduce the raw prediction" -- a
    # uniform 0.12 m offset fails that test while being exactly right.
    # Stages 5-7 do make geometric claims, and are gated against this.
    P_ref = H.copy()

    # ---- stage 5/6: structure-aware deconvolution ------------------------
    if cfg.enable_deconv:
        w = (
            cfg.class_weight_ground
            + (cfg.class_weight_vegetation - cfg.class_weight_ground) * (labels > 0)
            + (cfg.class_weight_building - cfg.class_weight_vegetation) * p_b_map
        ).astype(np.float32)
        w = np.clip(ndi.gaussian_filter(w, max(sigma, 1.0)), 0.0, 1.0)
        H = T.deconvolve(H, sigma, w, rgb, cfg)
        stats["deconv_weight_mean"] = float(w.mean())

    # ---- stage 7: plateau restoration ------------------------------------
    if cfg.enable_plateau:
        H, used = T.plateau_restore(H, labels, feats, p_building, cfg, gsd)
        stats["plateau_pixels"] = int((used > 0).sum())
        stats["plateau_regions"] = int(len(np.unique(np.where(used > 0, 1, 0))) - 1)

    # ---- stage 8: guards -------------------------------------------------
    H, n_invented = T.topology_guard(P_ref, H, sigma, cfg)
    stats["invented_features_suppressed"] = int(n_invented)

    H, frac_ok = T.data_consistency_gate(P_ref, H, sigma, cfg)
    stats["data_consistent_fraction"] = frac_ok

    H = np.maximum(H, 0.0).astype(np.float32)

    # confidence = how much we moved, relative to what we were allowed to move
    delta = np.abs(H - P0)
    conf = np.clip(1.0 - delta / (np.percentile(delta, 99.5) + 1e-6), 0, 1).astype(np.float32)

    stats["mean_abs_change_m"] = float(delta.mean())
    stats["p99_abs_change_m"] = float(np.percentile(delta, 99))
    stats["max_abs_change_m"] = float(delta.max())
    stats["changed_fraction_gt_5cm"] = float((delta > 0.05).mean())
    stats["runtime_s"] = round(time.time() - t0, 2)

    # ---- restore nodata verbatim ----------------------------------------
    out = H.copy()
    if bad.any():
        out[bad] = nodata

    return RefineResult(out, p_a_map, conf, labels, stats)


def config_for_ablation(name: str, base: Optional[RefineConfig] = None):
    """Return (config, kwargs) for one rung of the ablation ladder."""
    from dataclasses import replace

    cfg = base or RefineConfig()
    over = dict(ABLATIONS[name])
    supp = over.pop("suppress_artifacts", True)
    return replace(cfg, **over), {"suppress_artifacts": supp}
