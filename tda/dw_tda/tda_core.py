"""Backwards-compatible facade over the split modules.

The refinement used to live in one 755-line file. It is now three parts that can
be owned and reviewed independently:

    config.py      RefineConfig, nodata handling
    topology.py    persistence, merge tree, segmentation, blur/ground estimation
    regions.py     per-region features and the building-likeness classifier
    refine_ops.py  deconvolution, plateau restore, guards, artifact channel

Everything is re-exported here, so `from dw_tda import tda_core as T` and every
`T.something` call site continues to work unchanged. Verified byte-identical on
the 12-tile evaluation after the split.
"""
from .config import *          # noqa: F401,F403
from .config import RefineConfig, NODATA, split_nodata          # noqa: F401
from .topology import *        # noqa: F401,F403
from .regions import *         # noqa: F401,F403
from .refine_ops import *      # noqa: F401,F403
