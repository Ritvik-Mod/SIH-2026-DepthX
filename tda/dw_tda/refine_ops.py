"""Deconvolution, plateau restoration, and the guards that bound it.

This is where the sharpening actually happens -- the ablation ladder attributes
100% of the sharpness gain to deconvolve(). The guards matter as much as the
operator: morphological bounds make it arithmetically impossible to emit a
height no neighbouring pixel supports, and the topology guard flattens any new
maximum the observation did not already support.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from .config import RefineConfig, NODATA
from .topology import persistence_simplify, h_maxima_mask

# stage 5/6: structure-aware deconvolution
# --------------------------------------------------------------------------


def _rgb_edge_terms(rgb: Optional[np.ndarray], k_rgb: float) -> Optional[tuple]:
    """Directional RGB gradient penalties for the diffusion prior.  RGB is used
    only to *block* diffusion across image edges; its intensity is never
    transferred into height, so a dark roof cannot become a low roof."""
    if rgb is None:
        return None
    g = rgb.astype(np.float32)
    terms = []
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        d = np.roll(g, shift, axis=axis) - g
        terms.append((np.sum(d * d, axis=2) / (k_rgb ** 2 + 1e-12)).astype(np.float32))
    return tuple(terms)


def structure_aware_prior(
    H: np.ndarray,
    rgb_terms: Optional[tuple],
    cfg: RefineConfig,
    steps: int,
) -> np.ndarray:
    """Anisotropic (Perona-Malik) diffusion whose conductance is reduced both
    by height gradient and by RGB gradient.  Flattens roof interiors, refuses
    to diffuse across a roof edge visible in either channel."""
    out = H.astype(np.float32).copy()
    kh2 = cfg.k_height_m ** 2 + 1e-12
    for _ in range(max(steps, 0)):
        upd = np.zeros_like(out)
        for i, (shift, axis) in enumerate(((1, 0), (-1, 0), (1, 1), (-1, 1))):
            d = np.roll(out, shift, axis=axis) - out
            pen = cfg.height_prior_weight * (d * d) / kh2
            if rgb_terms is not None:
                pen = pen + cfg.rgb_prior_weight * rgb_terms[i]
            c = 1.0 / (1.0 + pen)
            upd += c * d
        out += cfg.prior_dt * upd
    return out


def deconvolve(
    P: np.ndarray,
    sigma: float,
    weight: np.ndarray,
    rgb: Optional[np.ndarray],
    cfg: RefineConfig,
) -> np.ndarray:
    """Landweber / iterative back-projection against the measured blur, with a
    structure-aware prior between steps, and hard projections after each step.

    The forward model is  observed = Gaussian_sigma * truth,  which is the one
    systematic error the pipeline documents and which this scene reproduces
    (measured sigma ~1.8 px here).  Deconvolution therefore *recovers*
    information that is present in the observation; it does not invent it.  The
    projections make that guarantee explicit:

      * non-negativity (AGL >= 0),
      * |refined - observed| <= cap, cap tied to local relief,
      * per-pixel refinement weight from the region classification, so
        vegetation and ground are barely touched.
    """
    P = P.astype(np.float32)
    rgb_terms = _rgb_edge_terms(rgb, cfg.k_rgb)
    win4 = int(max(3, round(4 * sigma))) | 1
    relief = ndi.maximum_filter(P, size=win4) - ndi.minimum_filter(P, size=win4)
    cap = np.clip(cfg.deviation_cap_rel * relief, cfg.deviation_cap_min_m, cfg.deviation_cap_max_m)

    # Morphological deblur bounds. Blurring is a local weighted average, so the
    # true height at a pixel cannot lie outside the range the observation takes
    # within the blur support: the roof value that got averaged away is still
    # present as a nearby local maximum, and the ground value as a nearby
    # local minimum. Enforcing this kills deconvolution ringing (which would
    # otherwise build a fake rim above every roof edge) and makes it
    # arithmetically impossible to invent a height no neighbouring pixel
    # supports.
    winb = int(max(3, round(cfg.bound_window_sigma * sigma))) | 1
    lo_b = ndi.minimum_filter(P, size=winb)
    hi_b = ndi.maximum_filter(P, size=winb)
    lo = np.maximum(np.maximum(P - cap, lo_b - cfg.bound_slack_m), 0.0)
    hi = np.minimum(P + cap, hi_b + cfg.bound_slack_m)

    X = P.copy()
    for _ in range(cfg.deconv_iters):
        resid = P - ndi.gaussian_filter(X, sigma, mode="nearest")
        X = X + cfg.deconv_step * ndi.gaussian_filter(resid, sigma, mode="nearest")
        X = structure_aware_prior(X, rgb_terms, cfg, cfg.prior_steps)
        X = np.clip(X, lo, hi)
    return (P + weight * (X - P)).astype(np.float32)


# --------------------------------------------------------------------------
# stage 7: plateau restoration
# --------------------------------------------------------------------------


def plateau_restore(
    H: np.ndarray,
    labels: np.ndarray,
    feats: Dict[str, np.ndarray],
    p_building: np.ndarray,
    cfg: RefineConfig,
    gsd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace a domed roof by its robust plane fit, blended by confidence.

    Only fires where the region is confidently building-like, big enough for a
    plane fit to mean anything, already close to planar, and shows the doming
    signature (height correlated with distance to the region edge) that a
    blurred plateau produces.  Returns (field, per-pixel blend actually used).
    """
    out = H.astype(np.float32).copy()
    used = np.zeros(H.shape, dtype=np.float32)
    if labels.max() == 0:
        return out, used
    objs = ndi.find_objects(labels)
    dt = ndi.distance_transform_edt(labels > 0)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        j = i - 1
        if p_building[j] < cfg.plateau_min_p_building:
            continue
        if feats["area_m2"][j] < cfg.plateau_min_area_m2:
            continue
        if feats["planarity"][j] > cfg.plateau_max_planarity_m:
            continue
        if feats["dome_corr"][j] < cfg.plateau_min_dome_corr:
            continue
        sub = labels[sl] == i
        core = sub & (dt[sl] > max(1.0, 1.5))
        if core.sum() < 12:
            continue
        ys, xs = np.nonzero(core)
        hs = H[sl][core]
        A = np.c_[xs, ys, np.ones(len(xs))]
        coef, *_ = np.linalg.lstsq(A, hs, rcond=None)
        # one robust reweighting pass so a chimney does not tilt the roof
        r = hs - A @ coef
        w = 1.0 / (1.0 + (r / (1.4826 * np.median(np.abs(r)) + 1e-3)) ** 2)
        coef, *_ = np.linalg.lstsq(A * w[:, None], hs * w, rcond=None)

        # Apply only to the core. The outer ring of a merge-tree region is
        # exactly where the blurred transition lives; pushing a roof plane out
        # to there moves the footprint edge, which is the frontend's silhouette.
        ys2, xs2 = np.nonzero(core)
        fit = coef[0] * xs2 + coef[1] * ys2 + coef[2]
        blend = float(np.clip(cfg.plateau_blend_max * (p_building[j] - cfg.plateau_min_p_building)
                              / max(1 - cfg.plateau_min_p_building, 1e-6), 0, cfg.plateau_blend_max))
        tgt = out[sl]
        cur = tgt[core]
        tgt[core] = (1 - blend) * cur + blend * np.maximum(fit, 0.0)
        u = used[sl]
        u[core] = blend
    return out, used


# --------------------------------------------------------------------------
# stage 8: topology guard + data consistency
# --------------------------------------------------------------------------


def topology_guard(
    P: np.ndarray, X: np.ndarray, sigma: float, cfg: RefineConfig
) -> Tuple[np.ndarray, int]:
    """Cap any structure the refinement invented.

    A maximum in the refined field with persistence >= tau_guard is legitimate
    only if the *observed* field already had a maximum of at least half that
    persistence nearby (within the blur radius, since blur is what displaces
    them).  Anything else is a feature we manufactured; it is flattened back to
    its merge saddle by local persistence simplification.  Returns the guarded
    field and the number of invented components suppressed.
    """
    r = cfg.guard_search_radius_px or max(2.0 * sigma, 2.0)
    new = h_maxima_mask(X, cfg.tau_guard_m)
    old = h_maxima_mask(P, 0.5 * cfg.tau_guard_m)
    support = ndi.binary_dilation(old, morphology.disk(int(math.ceil(r))))
    lab = measure.label(new)
    bad = np.zeros_like(new)
    n_bad = 0
    for i in range(1, lab.max() + 1):
        m = lab == i
        if not (m & support).any():
            bad |= m
            n_bad += 1
    if n_bad == 0:
        return X, 0
    bad = ndi.binary_dilation(bad, morphology.disk(int(math.ceil(2 * r))))
    flat = persistence_simplify(X, cfg.tau_guard_m)
    return np.where(bad, np.minimum(X, flat), X).astype(np.float32), n_bad


def data_consistency_gate(
    P: np.ndarray, X: np.ndarray, sigma: float, cfg: RefineConfig, block: int = 64
) -> Tuple[np.ndarray, float]:
    """Reject the refinement wherever it fails to explain the observation.

    The refinement claims  P ~= blur(X).  Compute the residual of that claim
    per block; where re-blurring the refined field fits the observed prediction
    worse than the observed prediction fits itself (times a tolerance), revert
    that block to the input.  This is the single most important safety
    property: it makes it structurally impossible for the method to "improve"
    a region by inventing geometry inconsistent with the model output.
    """
    if not cfg.require_data_consistency:
        return X, 1.0
    rx = (P - ndi.gaussian_filter(X, sigma, mode="nearest")) ** 2
    rp = (P - ndi.gaussian_filter(P, sigma, mode="nearest")) ** 2
    kx = ndi.uniform_filter(rx, block)
    kp = ndi.uniform_filter(rp, block)
    ok = kx <= cfg.consistency_tolerance * np.maximum(kp, 1e-8)
    frac = float(ok.mean())
    return np.where(ok, X, P).astype(np.float32), frac


# --------------------------------------------------------------------------
# dense artifact channel
# --------------------------------------------------------------------------


def artifact_probability(
    H: np.ndarray, rgb: Optional[np.ndarray], cfg: RefineConfig, gsd: float
) -> np.ndarray:
    """Per-pixel probability that a small-scale bump is a model artifact.

    Region-level classification cannot see these: a 2 px blister on a roof is
    below any sane minimum region size.  So this channel is dense and works on
    two signals only, both of which were checked against the data rather than
    assumed:

    * **scale** — how much of the height at this pixel disappears under a
      grey opening at ``artifact_scale_px``.  Real structures at 0.33 m/px
      (a shed, a car, a dormer) survive an opening at 3 px = 1 m; a 2-pixel
      blister does not.
    * **image support** — whether the RGB texture has an edge here.  A real
      object has a visible outline.  This is used as *evidence against*
      suppression, never as a classifier: greenness was measured to be
      useless on this scene (leaf-off imagery, medians 0.031 vs 0.035 for
      low- and high-persistence maxima), so only gradient support is used.

    Both signals must agree before anything is suppressed, and the
    suppression itself is bounded by ``artifact_max_persistence_m``.
    """
    if not cfg.enable_artifact_suppression:
        return np.zeros(H.shape, dtype=np.float32)
    k = max(int(cfg.artifact_scale_px) | 1, 3)
    small_scale = H - ndi.grey_opening(H, size=k)
    s = np.clip(small_scale / max(cfg.artifact_max_persistence_m, 1e-6), 0, 1)

    if rgb is not None:
        grey = rgb.mean(axis=2).astype(np.float32)
        gy, gx = np.gradient(ndi.gaussian_filter(grey, 1.0))
        cgm = ndi.maximum_filter(np.hypot(gx, gy), k)
        ref = np.percentile(cgm, cfg.rgb_support_percentile) + 1e-9
        support = np.clip(cgm / ref, 0, 1)
    else:
        support = np.zeros(H.shape, dtype=np.float32)

    return (s * (1.0 - support)).astype(np.float32)


def suppress_artifacts(
    H: np.ndarray, p_art: np.ndarray, cfg: RefineConfig
) -> np.ndarray:
    """Flatten suspected artifacts to their merge saddle, blended by
    confidence.  Implemented as persistence simplification, so a bump with
    persistence above the threshold is untouched no matter how suspicious it
    looks — a 1.5 m car cannot be erased by a 0.45 m operator."""
    if not p_art.any():
        return H
    flat = persistence_simplify(H, cfg.artifact_max_persistence_m)
    return ((1 - p_art) * H + p_art * np.minimum(H, flat)).astype(np.float32)
