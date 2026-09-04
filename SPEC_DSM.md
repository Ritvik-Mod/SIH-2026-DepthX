# Output contract — AGL → DSM stage

**Owner:** Divyanshu · **Consumes:** the height stage's `foo.tif` (see [`SPEC.md`](SPEC.md))
**Consumers:** Aakarsh (3D viewer), Vineeth (TDA), the evaluator
**Covers:** PS requirements 2–6 and 13

---

## What this stage does, in one line

The height stage answers *"how tall is the thing at this pixel above the ground directly
beneath it"*. This stage answers *"how high is the top of it above sea level"*.

```
DSM  =  DTM  +  AGL
        ↑        ↑
     terrain   from the height stage
```

A 6-storey building on a 200 m hill: AGL says `18`, this stage says `218`.

> ⚠️ **Terrain is ADDED.** `DTM − AGL` produces a scene that looks like a plausible
> landscape, renders fine, and is completely wrong. `assert_dsm_above_dtm()` runs on
> every write specifically to catch it, because nothing else does.

---

## Run it

```bash
# simplest thing that works — no downloads, no extra data
python -m heightmap.to_dsm --agl outputs/contract_sample/contract_sample.tif \
                           --out outputs/dsm --base-elevation 12

# no spatial metadata anywhere → a relative surface, honestly labelled
python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm --mode relative

# you have a DEM file covering the area
python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm --dem cop30.tif

# you have surveyed points — a CSV of  x,y,elevation
python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm --gcp points.csv

# fix the whole-scene offset using the land-cover map
python -m heightmap.to_dsm --agl scene.tif --out outputs/dsm \
                           --base-elevation 216 --semantic scene_cls.tif
```

`python tests/test_dsm.py` exercises every path above with no data at all.

---

## The two modes (PS requirements 5 and 6)

Chosen automatically from what the input carries. `--mode` forces either.

| Input | Mode | `quantity` | `units` | Range |
|---|---|---|---|---|
| GeoTIFF with CRS + transform, **or** any input with a terrain source | **absolute** | `DSM` | `metres` | real elevations |
| plain PNG/JPG — no location, no scale | **relative** | `rDSM` | `relative` | `[0, 1]` |

Relative mode does not invent metres. Without a location there is no way to look up a
ground elevation, and writing metres you cannot justify is worse than writing an honest
relative surface. It is still directly usable for 3D displacement.

`rDSM` uses **1st/99th percentiles, not min/max**. One 390 m outlier — GAMUS has such a
tile — would otherwise push every building to the bottom of the range and render the
whole scene flat.

---

## Files

For an input AGL raster `foo.tif`:

| File | Contents |
|---|---|
| `foo_dsm.tif` / `foo_rdsm.tif` | **the deliverable** — float32, 1 band |
| `foo_dsm.json` | sidecar: datum, terrain source, calibration, assumptions |
| `foo_dsm_preview.png` | colour-mapped, for humans. **Never** read values from it |

Raster: float32, 1 band, nodata `-9999.0`, DEFLATE, internally tiled 256×256, CRS and
transform written when the input was georeferenced. Key facts are also written as GDAL
tags (`quantity`, `units`, `vertical_datum`, `dtm_source`) so `gdalinfo` alone tells you
what a file is.

### Sidecar

```json
{
  "schema_version": "1.0",
  "quantity": "DSM",
  "units": "metres",
  "vertical_datum": "EGM2008 (orthometric, approx. mean sea level)",
  "georeferenced": true,
  "crs": "EPSG:32643",
  "transform": [0.33, 0.0, 700000.0, 0.0, -0.33, 3160000.0],
  "pixel_spacing_m": 0.33,
  "dtm_source": "dem_file | constant_plane | gcp_fit | none",
  "dtm_method": "morphological opening, 120 m window",
  "dtm_details": {"window_m": 150.0, "mean_height_removed_m": 4.04, "resampling": "cubic"},
  "calibration": {"scale": 1.0, "offset": -2.02, "n_reference_points": 906976,
                  "fit": "offset only", "reference": "self: ground+road pixels assumed 0 m AGL"},
  "agl_source": "a7_vitl_v0.1.0.pt",
  "value_range": {"min": 201.44, "max": 236.45, "p1": 201.5, "p50": 203.47, "p99": 233.1},
  "terrain_range": {"min": 201.45, "max": 205.31, "mean": 203.2},
  "nodata": -9999.0,
  "assumptions": ["terrain treated as flat across the tile at 12 m"]
}
```

**`assumptions` is not decoration.** Every shortcut taken goes in it. A stated
assumption is engineering; an unstated one is a bug waiting for someone else to find.

---

## Where the terrain comes from

Most trustworthy first. Pick one.

| Flag | Strategy | Good for | Accuracy |
|---|---|---|---|
| `--gcp points.csv` | tilted plane through surveyed points | you have survey data | ±0.5 m at the points |
| `--dem file.tif` | strip buildings off a DEM, resample to our grid | large scenes, any relief | ±3–8 m |
| `--base-elevation M` | one flat plane | a single tile up to ~1 km | ±1–2 m of terrain error |
| *(nothing)* | flat plane at 0 m | shape only; every value offset | — |

### The trap: free global elevation data is not bare earth

SRTM and Copernicus GLO-30 are **DSMs, not DTMs** — ESA's own description of Copernicus
says it includes buildings, infrastructure and vegetation. Over a city SRTM already sits
at roughly rooftop level, so:

```
WRONG:  SRTM + AGL  =  rooftop + 18 m  =  a 36 m building.  Every building doubled.
```

Nothing errors and the render looks fine. So `--dem` **never** uses the file directly:
it runs `dem_to_dtm()` first, a morphological opening that erases anything narrower than
the window and standing above its surroundings, and keeps everything broader.

The window must be **wider than the widest building** or the middle of a large flat roof
survives and becomes permanent terrain. Default 120 m; a mall or airport terminal needs
more (`--max-building-width`). Too wide is also wrong — it flattens genuine hills. The
run prints `mean_height_removed_m`; if that is ~0 over a city, your window is too narrow.

Over bare mountains the trap does not exist — nothing stands above the ground, so
DSM ≈ DTM and the DEM *is* the answer. The filter is harmless there.

Resampling from 30 m to 0.33 m is **cubic, never nearest** — nearest gives a staircase
under sharp buildings, and the steps read as terrain features that do not exist.

---

## Calibration (PS requirement 3)

The model already outputs metres, so this is residual correction, not scale invention.
There is measured evidence it is needed: on the held-out New York test the model kept
building heights almost perfectly (1.585 m MAE, r = 0.901) while the whole scene sat
**2.02 m too low**. A near-perfect shape with a constant offset is exactly what a linear
fit removes.

**Fit on ground, not on buildings.** Building heights are the part that is already
right; fitting there spends the fit on the good half and leaves the offset in place.
Ground is class `1` and road is class `5`.

With `--semantic`, no external data is needed at all: ground and road are ~0 m AGL *by
definition*, so whatever the model predicts there is its bias. Below 50 reference
samples a free scale factor fits noise, so `calibrate()` drops to offset-only
automatically and records the downgrade in the sidecar.

---

## Vertical datums — the tens-of-metres trap

"Height above sea level" is not one number.

| Family | Zero means | Examples |
|---|---|---|
| **Ellipsoidal** | a smooth mathematical model of the Earth | raw GPS, WGS84 |
| **Orthometric** | the geoid — real mean sea level, which is lumpy | SRTM (EGM96), Copernicus (EGM2008) |

The gap runs from about −100 m to +85 m worldwide; **over India roughly −30 m to −70 m.**
Mix an ellipsoidal reference with an orthometric DEM and every elevation is tens of
metres wrong, with no error, no warning, and a perfectly normal-looking render.

1. Every file this stage writes states its `vertical_datum`, in the sidecar and as a
   GDAL tag.
2. Read the datum of every file you consume. Never assume. The run warns when a DEM
   declares none.
3. Converting needs a geoid model — `pyproj` with the right pipeline. Do not hand-code
   an offset.

---

## Metrics (PS requirement 13)

`heightmap.dsm.metrics()` reports **RMSE, MAE, correlation and signed bias**, pooled and
split by land cover.

Split, because roughly a third of pixels are ground sitting at almost exactly 0 m and
are nearly free to get right — a pooled number flatters you.

Signed, because two models can both score 4 m MAE: one randomly off by ±4 m, the other
consistently 4 m short. Only the sign separates them, and a consistent offset is the
fixable kind.

---

## Mistakes this stage is built to prevent

| Mistake | What happens | Guard |
|---|---|---|
| Subtracting terrain | plausible, completely wrong scene | `assert_dsm_above_dtm()`, on every write |
| SRTM/GLO-30 used directly as DTM | every building doubled | `--dem` always runs the opening filter |
| Forgetting nodata | a mean shifts by kilometres | `-9999` masked before any statistic |
| Mixing vertical datums | silent 30–70 m error over India | datum recorded and read on every file |
| Nearest-neighbour DEM upsampling | staircase terrain under buildings | cubic, then smoothed |
| Reading `_preview.png` | 8-bit percentile-stretched; heights become fiction | `read_agl()` refuses PNG input |
| Calibrating on building pixels | fits the half that is already right | ground + road only |
| Opening window too narrow | large roofs become permanent hills | `mean_height_removed_m` printed |
| Feeding a DSM back in | terrain added twice | sidecar `quantity` checked on read |

---

## Still to do

| # | Task | Needs |
|---|---|---|
| 1 | Auto-fetch Copernicus GLO-30 for a scene's bounds | network |
| 2 | Geoid conversion via `pyproj` when datums differ | — |
| 3 | Validate `DSM = DTM + AGL` on US3D, which ships both | the US3D download |

US3D (DFC2019) is the dataset that lets you check the whole chain: Track 1 gives AGL,
Track 3 gives true LiDAR DSM on the same tile ids, so `DTM_true = DSM_true − AGL_true`
is derivable and `dem_to_dtm()` can be tuned against ground truth instead of guessed.
`heightmap/data/us3d.py` and `scripts/inspect_us3d.py` already handle the loading.
