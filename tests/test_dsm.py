"""Tests for the AGL -> DSM stage.

Every test here exists because the corresponding mistake is silent: it produces a file
that opens, renders, and is wrong.  Assertions are the only thing standing between this
stage and a plausible-looking disaster, so they are deliberately specific.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from heightmap.dsm import (NODATA, to_dsm, to_rdsm, assert_dsm_above_dtm, constant_plane,
                           gcp_plane, dem_to_dtm, calibrate, calibrate_to_flat_ground,
                           metrics)
from heightmap.data.classes import GROUND, BUILDING, ROAD


def scene(h=64, w=64, height=18.0):
    """A tile with one building on flat ground, plus its land-cover map."""
    agl = np.zeros((h, w), np.float32)
    agl[20:44, 20:44] = height
    sem = np.where(agl > 1, BUILDING, GROUND).astype(np.uint8)
    return agl, sem, np.ones((h, w), bool)


def test_addition():
    """DSM = DTM + AGL, exactly."""
    agl, _, valid = scene()
    dtm, info = constant_plane(agl.shape, 200.0)
    dsm = to_dsm(agl, dtm, valid)
    assert info["dtm_source"] == "constant_plane"
    assert np.isclose(dsm[0, 0], 200.0), "ground must read the terrain elevation"
    assert np.isclose(dsm[30, 30], 218.0), "a 18 m building on 200 m ground reads 218 m"
    assert np.allclose(dsm - dtm, agl), "the difference must be the AGL we put in"
    print(f"addition:   ground {dsm[0,0]:.1f} m   roof {dsm[30,30]:.1f} m   OK")


def test_subtraction_is_caught():
    """The one error nothing else catches: terrain subtracted instead of added."""
    agl, _, valid = scene()
    dtm, _ = constant_plane(agl.shape, 200.0)

    wrong = dtm - agl                       # the mistake, in full
    assert wrong.min() > 0 and wrong.max() < 300, \
        "the wrong answer is in a plausible range -- which is exactly why it is dangerous"
    for kwargs in ({}, {"agl": agl}):       # with and without the AGL cross-check
        try:
            assert_dsm_above_dtm(wrong, dtm, valid, **kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"subtraction was NOT caught (kwargs={list(kwargs)})")
    print(f"subtraction: caught (wrong answer spans {wrong.min():.0f}-{wrong.max():.0f} m, "
          f"looks perfectly plausible)")


def test_calibration_noise_is_not_caught():
    """Ground a few cm negative after calibration is the calibration working, not a bug.

    A check tight enough to reject this would fire on every real scene and get switched
    off, taking the subtraction check with it.
    """
    agl, _, valid = scene()
    agl = agl - 0.02                                    # what offset calibration leaves
    dtm, _ = constant_plane(agl.shape, 200.0)
    dsm = to_dsm(agl, dtm, valid)                       # must not raise
    assert np.isclose(dsm[0, 0], 199.98)
    print("cm-scale negative ground after calibration: accepted, as it must be")


def test_rdsm_survives_an_outlier():
    """Robust percentiles, not min/max: one wild tile must not flatten the scene."""
    agl, _, valid = scene(height=18.0)
    agl[0, 0] = 390.0                                   # GAMUS has a tile like this

    r, norm = to_rdsm(agl, valid)
    minmax = (agl - agl.min()) / (agl.max() - agl.min())

    assert 0.0 <= r.min() and r.max() <= 1.0, "rDSM must stay inside [0, 1]"
    assert r[30, 30] > 0.9, "the building must still use most of the range"
    assert minmax[30, 30] < 0.06, "min/max normalisation flattens it -- the bug we avoid"
    print(f"rDSM:       building at {r[30,30]:.2f} of range "
          f"(min/max would give {minmax[30,30]:.3f})   lo={norm['lo']:.1f} hi={norm['hi']:.1f}")


def test_nodata_never_reaches_a_statistic():
    """-9999 is a sentinel, not a height.  Averaged in, it shifts a mean by kilometres."""
    agl, _, valid = scene()
    agl[0, :] = NODATA
    valid[0, :] = False

    r, _ = to_rdsm(agl, valid)
    assert r[0, 0] == 0.0 and r.min() >= 0.0, "nodata must not leak into the output"

    dtm, _ = constant_plane(agl.shape, 200.0)
    dsm = to_dsm(np.where(valid, agl, np.nan), dtm, valid)
    finite = dsm[np.isfinite(dsm)]
    assert finite.min() >= 199.9, f"nodata leaked into the DSM: min {finite.min()}"
    print(f"nodata:     masked; DSM min {finite.min():.1f} m (not -9799)")


def test_dem_opening_strips_buildings():
    """A DEM is a surface model.  Used raw as a DTM it double-counts every building."""
    n, px = 40, 30.0
    yy, xx = np.mgrid[0:n, 0:n]
    terrain = 200.0 + 0.25 * xx                          # a gentle real slope
    dem = terrain.copy()
    dem[10:13, 10:13] += 25.0                            # 90 m wide buildings, 25 m tall

    dtm, info = dem_to_dtm(dem, px, max_building_width_m=150)
    err_building = float(np.abs(dtm[10:13, 10:13] - terrain[10:13, 10:13]).max())
    err_slope = float(np.abs(dtm[20:30, 20:30] - terrain[20:30, 20:30]).max())

    assert err_building < 3.0, f"buildings survived the opening ({err_building:.1f} m left)"
    assert err_slope < 3.0, f"the real slope was flattened ({err_slope:.1f} m error)"
    print(f"dem->dtm:   raw DEM was {dem[11,11]-terrain[11,11]:.0f} m too high over the "
          f"building; after opening {err_building:.2f} m. Slope preserved to {err_slope:.2f} m")


def test_dem_window_too_narrow_leaves_buildings():
    """The documented failure: a window narrower than the building keeps its roof."""
    n, px = 40, 30.0
    dem = np.full((n, n), 200.0)
    dem[8:20, 8:20] += 25.0                              # 360 m wide
    narrow, _ = dem_to_dtm(dem, px, max_building_width_m=120)
    assert narrow[13, 13] > 220.0, "expected the roof to survive a too-narrow window"
    wide, _ = dem_to_dtm(dem, px, max_building_width_m=420)
    assert wide[13, 13] < 206.0, "a wide enough window should erase it"
    print(f"window:     120 m window leaves {narrow[13,13]-200:.0f} m of roof as 'terrain'; "
          f"420 m window leaves {wide[13,13]-200:.1f} m")


def test_calibration_recovers_a_known_offset():
    """The New York case: shape right, whole scene 2.02 m low."""
    agl, sem, valid = scene()
    biased = agl - 2.02

    fixed, rep = calibrate_to_flat_ground(biased, sem, valid=valid)
    assert abs(rep["offset"] - 2.02) < 1e-3, f"offset came back as {rep['offset']}"
    assert abs(float(np.mean(fixed[sem == GROUND]))) < 1e-3
    assert abs(float(np.mean(fixed[sem == BUILDING])) - 18.0) < 1e-3, \
        "calibration must not disturb the building heights, which were already right"
    print(f"calibrate:  recovered {rep['offset']:+.3f} m from {rep['n_reference_points']:,} "
          f"ground pixels; buildings untouched at "
          f"{float(np.mean(fixed[sem==BUILDING])):.2f} m")


def test_calibration_uses_road_as_ground_too():
    agl, sem, valid = scene()
    sem[0:10, :] = ROAD
    _, rep = calibrate_to_flat_ground(agl - 1.5, sem, valid=valid)
    assert rep["n_reference_points"] > (sem == GROUND).sum(), "road pixels must be included"
    print(f"road class: included ({rep['n_reference_points']:,} reference pixels)")


def test_few_samples_refuse_a_free_scale():
    """With a handful of points a free scale factor fits noise.  It must downgrade."""
    rng = np.random.default_rng(0)
    agl = rng.random((10, 10)).astype(np.float32) * 20
    mask = np.zeros((10, 10), bool)
    mask[0, :5] = True                                    # 5 samples
    _, rep = calibrate(agl, agl + 3.0, mask, fit_scale=True)
    assert rep["fit"] == "offset only", "should have refused to fit a scale"
    assert abs(rep["offset"] - 3.0) < 1e-4
    assert "note" in rep, "the downgrade must be recorded, not silent"
    print(f"few points: {rep['fit']}, offset {rep['offset']:+.3f} m -- downgrade recorded")


def test_gcp_plane_recovers_a_tilt():
    """Three or more points give tilt as well as offset."""
    h = w = 32
    cols, rows = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    truth = 0.5 * cols - 0.25 * rows + 100.0
    xs = [0.0, 31.0, 0.0, 31.0]
    ys = [0.0, 0.0, 31.0, 31.0]
    zs = [0.5 * x - 0.25 * y + 100.0 for x, y in zip(xs, ys)]

    dtm, info = gcp_plane((h, w), xs, ys, zs)
    assert info["n_points"] == 4 and info["rms_residual_m"] < 1e-6
    assert np.abs(dtm - truth).max() < 1e-3, "the fitted plane should reproduce the truth"

    flat, finfo = gcp_plane((h, w), [0.0, 31.0], [0.0, 0.0], [100.0, 110.0])
    assert "too few" in finfo["method"], "2 points must not extrapolate a tilt"
    assert np.allclose(flat, 105.0)
    print(f"gcp:        4 points -> tilt recovered (rms {info['rms_residual_m']:.1e} m); "
          f"2 points -> flat at {flat[0,0]:.0f} m, no wild extrapolation")


def test_metrics_report_signed_bias_per_class():
    agl, sem, valid = scene()
    r = metrics(agl, agl, valid, sem)
    assert r["all"]["mae_m"] == 0.0 and r["all"]["rmse_m"] == 0.0

    short = agl - 4.0
    r2 = metrics(short, agl, valid, sem)
    assert abs(r2["all"]["bias_m"] + 4.0) < 1e-6, "signed bias must show the direction"
    assert r2["building"]["n"] == int((sem == BUILDING).sum())
    assert r2["ground"]["n"] == int((sem == GROUND).sum())
    print(f"metrics:    uniformly 4 m short -> bias {r2['all']['bias_m']:+.2f} m, "
          f"MAE {r2['all']['mae_m']:.2f} m; ground/building split reported separately")


def main():
    for fn in (test_addition, test_subtraction_is_caught, test_calibration_noise_is_not_caught,
               test_rdsm_survives_an_outlier, test_nodata_never_reaches_a_statistic,
               test_dem_opening_strips_buildings, test_dem_window_too_narrow_leaves_buildings,
               test_calibration_recovers_a_known_offset, test_calibration_uses_road_as_ground_too,
               test_few_samples_refuse_a_free_scale, test_gcp_plane_recovers_a_tilt,
               test_metrics_report_signed_bias_per_class):
        fn()
    print("\nall DSM tests passed")


if __name__ == "__main__":
    main()
