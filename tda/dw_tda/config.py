"""Refinement configuration and nodata handling.

Split out of the original single-file tda_core so the three parts of the method
-- configuration, topology, and the refinement operators -- can be owned and
reviewed separately. Behaviour is unchanged; tda_core re-exports everything.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@dataclass
class RefineConfig:
    """Every threshold the method uses. Serialised with each run."""

    # --- stage 1: micro-ripple persistence simplification -----------------
    tau_micro_m: float = 0.08
    """Maxima with persistence below this are numerical ripple, not structure.
    0.08 m is below airborne-LiDAR vertical noise, so no removed feature could
    have been verified against the ground truth anyway.  Bounded damage: this
    stage can move no pixel by more than tau_micro_m."""

    # --- stage 2: AGL zero-level calibration ------------------------------
    ground_window_m: float = 45.0
    """Window for the local-ground estimate.  Must exceed the largest building
    footprint in the scene, otherwise a big flat roof is mistaken for ground."""
    ground_quantile: float = 0.02
    ground_max_correction_m: float = 0.60
    """Hard cap. The zero-level fix is a definitional correction of a few
    centimetres; if the estimate exceeds this the scene probably has real
    elevated terrain and we refuse to touch it."""
    enable_ground_zeroing: bool = True

    # --- stage 3: merge-tree segmentation ---------------------------------
    tau_seg_m: float = 0.60
    """Persistence a maximum needs to seed its own region.  Below this, two
    adjacent roof planes are treated as one structure."""
    elevated_floor_m: float = 1.20
    """Height above which a pixel can belong to an elevated region."""
    min_region_area_m2: float = 4.0

    # --- stage 4: region classification -----------------------------------
    # logistic weights on standardised features -> p(building-like).
    # Defaults are hand-set from the observed statistics of GAMUS-style urban
    # tiles; `evaluate_batch.py --fit-classifier` refits them from GT pairs.
    cls_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "bias": -0.35,
            "z_log_area": 0.95,
            "z_planarity": -1.25,
            "z_rgb_texture": -0.85,
            "z_solidity": 0.70,
            "z_rectangularity": 0.55,
            "z_edge_alignment": 0.80,
            "z_greenness": -0.30,
            "z_persistence": 0.45,
        }
    )
    artifact_max_area_m2: float = 3.0
    artifact_max_persistence_m: float = 0.45
    artifact_min_edge_alignment: float = 0.18
    min_region_area_m2_note: str = "regions below this are handled by the dense artifact channel"

    # --- stage 5/6: structure-aware deconvolution -------------------------
    enable_deconv: bool = True
    sigma_override_px: Optional[float] = None
    sigma_min_px: float = 0.8
    sigma_max_px: float = 4.0
    deconv_iters: int = 45
    deconv_step: float = 1.0
    prior_steps: int = 3
    prior_dt: float = 0.18
    k_height_m: float = 0.45
    """Perona-Malik edge scale on the height field, in metres per pixel step."""
    k_rgb: float = 0.08
    """Edge scale on the RGB guide (0-1 intensity units per pixel step)."""
    rgb_prior_weight: float = 1.0
    height_prior_weight: float = 0.5
    class_weight_building: float = 1.00
    class_weight_vegetation: float = 0.30
    class_weight_ground: float = 0.15
    deviation_cap_rel: float = 0.60
    deviation_cap_min_m: float = 0.20
    deviation_cap_max_m: float = 6.0
    bound_window_sigma: float = 3.0
    """Radius, in units of sigma, of the morphological deblur bounds."""
    bound_slack_m: float = 0.05

    # --- artifact channel (small-scale, image-unsupported bumps) ----------
    enable_artifact_suppression: bool = True
    artifact_scale_px: int = 3
    rgb_support_percentile: float = 90.0

    # --- stage 7: plateau restoration -------------------------------------
    enable_plateau: bool = False
    """Off by default: measured worse than plain A4 on development data."""
    plateau_min_p_building: float = 0.62
    plateau_min_area_m2: float = 12.0
    plateau_max_planarity_m: float = 1.60
    plateau_min_dome_corr: float = 0.25
    plateau_blend_max: float = 0.75

    # --- stage 8: topology guard ------------------------------------------
    tau_guard_m: float = 0.50
    guard_search_radius_px: Optional[float] = None  # default 2*sigma

    # --- global safety ----------------------------------------------------
    require_data_consistency: bool = True
    consistency_tolerance: float = 1.0
    """A refinement is only accepted where re-blurring it reproduces the input
    at least this well relative to the input's own residual (<=1.0 means the
    refined field must explain the observation no worse than the input does)."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


NODATA = -9999.0


# --------------------------------------------------------------------------
# nodata handling
# --------------------------------------------------------------------------


def split_nodata(H: np.ndarray, nodata: float = NODATA) -> Tuple[np.ndarray, np.ndarray]:
    """Return (filled field, nodata mask).  Nodata is filled by nearest valid
    neighbour so that filters do not smear -9999 into real geometry; the mask
    is re-applied verbatim on output."""
    H = np.asarray(H, dtype=np.float32)
    bad = ~np.isfinite(H)
    if nodata is not None:
        bad |= np.isclose(H, nodata)
    if not bad.any():
        return H.copy(), bad
    if bad.all():
        raise ValueError("heightmap is entirely nodata")
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return H[tuple(idx)].astype(np.float32), bad


# --------------------------------------------------------------------------
