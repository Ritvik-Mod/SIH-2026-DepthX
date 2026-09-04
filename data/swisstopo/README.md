# swisstopo — 10 scenes over steep, built-up terrain

Fetched by [`scripts/download_swisstopo.py`](../../scripts/download_swisstopo.py).
Re-run that to reproduce or extend this set.

## Why this data exists

Every dataset in this project so far is **flat**: GAMUS is Washington DC / New York /
Philadelphia, US3D is Jacksonville and Omaha. The problem statement grades height
accuracy across *urban, sparse, **hilly** and forested* landscapes, and nothing in the
repo could measure the hilly case at all.

These scenes have **24 m to 115 m of terrain relief** with houses built on the slopes.

They also carry something no other dataset here has: **the DSM and the DTM for the same
ground.** That is what makes `AGL = DSM − DTM` real ground truth rather than an
assumption.

## What each scene contains

```
<name>/
  <name>_RGB.tif    3-band uint8   the top view          SWISSIMAGE dop10, 0.1 m source
  <name>_DSM.tif    float32        top of everything     swissSURFACE3D Raster, 0.5 m source
  <name>_DTM.tif    float32        bare earth            swissALTI3D, 0.5 m source
  <name>_AGL.tif    float32        derived: DSM − DTM    ground truth height above ground
  <name>.json       sidecar        terrain stats, datum, consistency check
```

All four rasters are **1024 × 1024 at 0.33 m/px** — a 338 × 338 m footprint, identical in
size and GSD to a GAMUS tile, so they drop into the existing pipeline unchanged. CRS is
**EPSG:2056** (Swiss LV95, units in metres). nodata is `-9999.0`.

0.33 m/px is deliberate: the model's trained GSD band is 0.165–0.65 m/px, and 0.33 is its
training resolution. The 0.1 m orthophoto is resampled down with cubic interpolation;
elevation uses bilinear, because cubic overshoots at building walls.

## The scenes

| Scene | Relief | Mean slope | Built | Ground agreement |
|---|---|---|---|---|
| stmoritz | 115 m | 16.2° | 49% | −0.043 m |
| weggis | 99 m | 14.7° | 32% | +0.045 m |
| baden | 96 m | 12.5° | 57% | +0.134 m |
| spiez | 88 m | 9.0° | 30% | −0.054 m |
| montreux | 46 m | 9.7° | 41% | −0.018 m |
| neuchatel | 36 m | 5.3° | 57% | +0.046 m |
| vevey | 35 m | 5.0° | 55% | +0.035 m |
| schaffhausen | 32 m | 7.7° | 57% | +0.002 m |
| sierre | 31 m | 10.3° | 34% | −0.091 m |
| zug | 24 m | 4.8° | 36% | +0.006 m |

## ⚠️ Vertical datum — read this before combining with anything

Elevation here is **LN02 (EPSG:5728)**, the Swiss national levelling network.

It is **not EGM2008**, which is what `heightmap/dsm/` defaults to, and it is not the
EGM96 that SRTM uses. Mixing datums without a geoid conversion is a silent multi-metre
error with no warning and a perfectly normal-looking render — the trap documented in
[`SPEC_DSM.md`](../../SPEC_DSM.md).

Pass it explicitly:

```bash
python -m heightmap.to_dsm --agl pred.tif --dem data/swisstopo/zug/zug_DTM.tif \
       --vertical-datum "LN02 (EPSG:5728)" --out outputs/dsm
```

## `DSM = DTM + AGL` — checked, for the first time in this project

This equation is asserted throughout the repo and had **never been verified against real
data**, because GAMUS ships no DTM. It can be verified here.

**Result: it holds.** Across all 10 scenes the two independently-produced swisstopo
products agree on open ground to within **±0.13 m**, and 8 of 10 agree within ±0.06 m.

Two honest caveats, both recorded per scene in `agl_consistency`:

1. **AGL is clipped at zero**, because nothing stands below the ground and GAMUS's AGL is
   also ≥ 0. So `DTM + AGL` reproduces `DSM` exactly *except* where the raw difference
   was negative. That clip is the only source of discrepancy.
2. **The raw difference goes slightly negative on 1–47% of pixels**, but almost entirely
   by centimetres — under 3% of pixels anywhere fall below −0.5 m. swissALTI3D and
   swissSURFACE3D come from separate acquisitions, so decimetre-scale disagreement is
   expected and is not an error in either product.

A note on how *not* to measure this: selecting "ground" by DTM slope does not work. The
DTM is smooth **underneath** buildings, so a flat roof on flat terrain passes a slope
test and the statistic ends up measuring rooftops — it read +3.9 m on Neuchâtel that way.
`ground_agreement_p25_m` uses the 25th percentile of `DSM − DTM` instead, which lands on
open ground without needing to classify anything.

## What this unlocks

- **The hilly landscape number**, which the PS grades and the project could not produce.
- **Tuning `dem_to_dtm()`** against real bare earth instead of the guessed 120 m window:
  run it on the DSM, compare to the true DTM.
- **A real test of the terrain path** — `--dem` with actual relief, not a flat plane.
- **A sensor and country shift** for Ritvik: different optics, different architecture,
  different vegetation from the US cities the model trained on.

## Licence and attribution

swisstopo **open government data** — free to use, including commercially, **with
attribution**. Cite as:

> Source: Federal Office of Topography swisstopo — SWISSIMAGE, swissALTI3D,
> swissSURFACE3D. https://www.swisstopo.admin.ch

Keep this attribution in any slide, report or demo that shows these scenes.
