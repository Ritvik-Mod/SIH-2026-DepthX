"""Persistence, merge-tree segmentation, blur estimation, AGL zero level.

This is the topology proper: superlevel merge tree, persistence simplification,
h-maxima seeding and the watershed that splits touching buildings. Note that
topology contributes NO sharpening -- measured at -0.1% on 40 unseen tiles. Its
job is to decide WHERE correction is allowed and to prove afterwards that
nothing was invented.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation

from .config import RefineConfig, NODATA, split_nodata

# topology
# --------------------------------------------------------------------------


def persistence_simplify(H: np.ndarray, tau: float) -> np.ndarray:
    """Remove every merge-tree branch with persistence < tau by capping it at
    its merge saddle.  Equivalent to grayscale reconstruction by dilation of
    (H - tau) under H.  Monotone: 0 <= H - out <= tau everywhere."""
    if tau <= 0:
        return H.copy()
    seed = (H - np.float32(tau)).astype(np.float32)
    out = morphology.reconstruction(seed, H.astype(np.float32), method="dilation")
    return np.minimum(out, H).astype(np.float32)


def h_maxima_mask(H: np.ndarray, tau: float) -> np.ndarray:
    """Regional maxima whose persistence (dynamics) is at least tau."""
    return morphology.h_maxima(H.astype(np.float32), max(tau, 1e-6)) > 0


def persistence_diagram(H: np.ndarray, min_persistence: float = 0.0) -> np.ndarray:
    """0-dim persistence diagram of the superlevel filtration, by union-find.

    Returns an (n, 2) array of (birth, death) with birth > death, i.e. the
    height at which a component appears and the height at which it merges into
    an older component.  The global maximum's death is the field minimum
    (essential class).  O(N log N); ~10 s on 1024^2 in pure Python, so this is
    for evaluation/diagnostics, not the per-scene refinement path.
    """
    H = np.asarray(H, dtype=np.float64)
    h, w = H.shape
    flat = H.ravel()
    order = np.argsort(-flat, kind="stable")
    parent = np.full(h * w, -1, dtype=np.int64)
    birth = {}
    pairs = []

    def find(x: int) -> int:
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    offs = (-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1)
    for idx in order:
        v = flat[idx]
        y, x = divmod(int(idx), w)
        roots = set()
        for d in offs:
            j = int(idx) + d
            if j < 0 or j >= h * w:
                continue
            jy, jx = divmod(j, w)
            if abs(jy - y) > 1 or abs(jx - x) > 1:
                continue
            if parent[j] != -1:
                roots.add(find(j))
        if not roots:
            parent[idx] = idx
            birth[int(idx)] = v
        else:
            roots = list(roots)
            keep = roots[int(np.argmax([birth[r] for r in roots]))]
            parent[idx] = keep
            for r in roots:
                if r == keep:
                    continue
                pairs.append((birth.pop(r), v))
                parent[r] = keep
    fmin = float(flat.min())
    for r, b in birth.items():
        pairs.append((b, fmin))
    dgm = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    if min_persistence > 0 and len(dgm):
        dgm = dgm[(dgm[:, 0] - dgm[:, 1]) >= min_persistence]
    return dgm


def betti_curve(H: np.ndarray, levels: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """Number of connected components of {H >= t} for each t in `levels`,
    after persistence-simplifying at tau so the curve counts structures rather
    than ripple.  Cheap, vectorised alternative to a full diagram."""
    Hs = persistence_simplify(H, tau)
    return np.array(
        [ndi.label(Hs >= t)[1] for t in levels], dtype=np.float64
    )


def elevated_segmentation(
    H: np.ndarray, cfg: RefineConfig, gsd: float
) -> np.ndarray:
    """Partition the elevated set into merge-tree regions.

    Watershed of -H seeded by maxima with persistence >= tau_seg, restricted to
    {H >= elevated_floor}.  Two row houses of different heights sharing a party
    wall get two labels; one flat roof does not get split by ripple.
    """
    Hs = persistence_simplify(H, cfg.tau_micro_m)
    seeds = measure.label(h_maxima_mask(Hs, cfg.tau_seg_m))
    mask = Hs >= cfg.elevated_floor_m
    if not mask.any():
        return np.zeros(H.shape, dtype=np.int32)
    lab = segmentation.watershed(-Hs, seeds, mask=mask)
    min_px = max(int(round(cfg.min_region_area_m2 / (gsd * gsd))), 4)
    counts = np.bincount(lab.ravel())
    drop = np.where(counts < min_px)[0]
    if len(drop):
        lab[np.isin(lab, drop)] = 0
    # relabel sequentially, NOT by connectivity: two row houses sharing a party
    # wall are one connected component but two merge-tree regions, and
    # measure.label would silently glue them back together.
    lab, _, _ = segmentation.relabel_sequential(lab)
    return lab.astype(np.int32)


# --------------------------------------------------------------------------
# blur estimation  (the forward model of the systematic error)
# --------------------------------------------------------------------------


def estimate_blur_sigma(H: np.ndarray, cfg: RefineConfig) -> float:
    """Estimate, in pixels, the Gaussian sigma that best explains the softness
    of the height steps in this scene.

    For an ideal step of amplitude A blurred by a Gaussian of width sigma, the
    peak slope is A / (sigma * sqrt(2*pi)).  Measuring A locally as the
    max-min over a window and the slope from the gradient magnitude inverts
    that relation.  Taking the median over strong, high-amplitude edges is
    robust to genuinely sloped surfaces.
    """
    if cfg.sigma_override_px is not None:
        return float(np.clip(cfg.sigma_override_px, cfg.sigma_min_px, cfg.sigma_max_px))
    gy, gx = np.gradient(H.astype(np.float32))
    gm = np.hypot(gx, gy)
    amp = ndi.maximum_filter(H, 15) - ndi.minimum_filter(H, 15)
    strong = (gm > np.percentile(gm, 99.0)) & (amp > 2.0)
    if strong.sum() < 200:
        strong = (gm > np.percentile(gm, 99.5)) & (amp > 0.8)
    if strong.sum() < 50:
        return float(np.clip(1.5, cfg.sigma_min_px, cfg.sigma_max_px))
    sig = amp[strong] / (gm[strong] * math.sqrt(2 * math.pi) + 1e-9)
    return float(np.clip(np.median(sig), cfg.sigma_min_px, cfg.sigma_max_px))


# --------------------------------------------------------------------------
# stage 2: AGL zero-level
# --------------------------------------------------------------------------


def estimate_local_ground(H: np.ndarray, cfg: RefineConfig, gsd: float) -> np.ndarray:
    """AGL is defined as zero at local ground.  Estimate the residual floor as
    a low quantile over a window wider than any building, smooth it, and clamp
    it to [0, ground_max_correction_m].  This is a definitional calibration of
    the zero level, not a rescaling: it is a bounded additive offset, applied
    identically to every pixel in a neighbourhood."""
    w = max(int(round(cfg.ground_window_m / gsd)) | 1, 5)
    # Grey-scale opening with a window wider than any building returns the
    # local floor without the spatial shrinkage a bare min-filter causes:
    # erosion drops to the floor, the matched dilation puts it back in place.
    g = ndi.grey_opening(H, size=w)
    g = ndi.uniform_filter(g, size=max(w // 4 | 1, 3))
    g = np.clip(g, 0.0, cfg.ground_max_correction_m)
    return g.astype(np.float32)


# --------------------------------------------------------------------------
