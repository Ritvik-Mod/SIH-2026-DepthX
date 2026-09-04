"""AGL -> DSM: scale calibration, terrain, and the absolute / relative output modes.

Owner: Divyanshu.  Covers problem-statement requirements 2-6 and 13.

The height model upstream answers "how tall is the thing at this pixel above the ground
directly beneath it".  A 6-storey building on a 200 m hill and the same building at sea
level both come out as 18 m.  That is correct and deliberate, but it is not what the
problem statement asks to be delivered -- it wants a DSM, the elevation of the top of
everything above sea level, so that building should read 218 m.

This package is that conversion:

    DSM = DTM + AGL

Upstream owns "the shape is right".  This stage owns "the units are right".
"""
from .convert import (NODATA, DEFAULT_VERTICAL_DATUM, read_agl, to_rdsm, to_dsm,
                      write_dsm, assert_dsm_above_dtm)
from .terrain import constant_plane, gcp_plane, dem_to_dtm, load_dem_as_dtm
from .calibrate import (calibrate, calibrate_to_flat_ground, ground_mask, metrics,
                        GROUND_CLASSES)

__all__ = [
    "NODATA", "DEFAULT_VERTICAL_DATUM", "read_agl", "to_rdsm", "to_dsm", "write_dsm",
    "assert_dsm_above_dtm", "constant_plane", "gcp_plane", "dem_to_dtm",
    "load_dem_as_dtm", "calibrate", "calibrate_to_flat_ground", "ground_mask",
    "metrics", "GROUND_CLASSES",
]
