# The DSM stage — everything we built, explained from zero

**Owner:** Divyanshu · **Built:** 4 September 2026 · **Branch:** `export-pipeline` → merged to `main`

This document assumes you know **nothing** about satellites, elevation data, or deep
learning. Every term is defined before it is used. It exists so you can understand your
own module and defend it to a judge.

If you want the terse engineering contract instead, read [`SPEC_DSM.md`](SPEC_DSM.md).

---

## Part 1 — The project, in plain words

### 1.1 The competition problem

**SIH26175, ISRO.** Take **one flat photograph** shot straight down from a satellite.
Work out **how tall everything in it is**. Let someone fly through the result in 3D in a
web browser.

That sounds impossible — a photo is flat. But there are real clues:

- **Shadows.** A tall building throws a long shadow. Measure the shadow, know where the
  sun was, and you can compute the height. This is the strongest clue.
- **Known sizes.** Cars are ~4.5 m long, road lanes ~3.5 m wide. Once you know how many
  metres one pixel covers, everything gets a scale.
- **Texture.** Roofs look different from roads. Trees look different from buildings.

The normal ways to measure height are expensive: two satellites photographing from
different angles, or laser scanning from an aircraft. Doing it from **one ordinary
photo** is far cheaper. That is the whole point.

### 1.2 Who does what on your team

| Person | Owns |
|---|---|
| **Ritvik** | Photo → how tall everything is (AGL), in metres |
| **Divyanshu (you)** | Turning that into real elevations — adding terrain, calibrating |
| **Vineeth** | Cleaning up noisy regions using topology (TDA) |
| **Aakarsh** | The 3D flythrough in the browser |

The split follows a real principle:

> **Ritvik owns "the shape is right." You own "the units are right."**

Those are genuinely different problems, and there is hard evidence they separate
cleanly. On the held-out New York test, Ritvik's model got building heights almost
perfectly (**1.585 m** average error) but the **whole scene sat 2.02 m too low**. The
shape was fine; the units drifted. That constant 2 m offset is *your* problem, and it is
the fixable kind.

---

## Part 2 — The three words everything depends on

Imagine a **6-storey building standing on a 200-metre hill.**

| Word | Full name | What it means | Our building |
|---|---|---|---|
| **DTM** | Digital **Terrain** Model | The bare ground, as if you bulldozed every building and tree | **200 m** |
| **AGL** | Above Ground Level | How tall the thing is above that ground | **18 m** |
| **DSM** | Digital **Surface** Model | Elevation of the *top* of everything | **218 m** |

The relationship is just addition:

```
   DSM   =   DTM   +   AGL
   218   =   200   +    18
    ↑         ↑          ↑
 what the   you       Ritvik's
 judges    supply      model
  want                 gives
                       you
```

**Ritvik's model gives you the 18. You supply the 200. You add.**

That is your entire job, stated completely. The addition is trivial. Getting a
trustworthy 200 is the hard part.

### ⚠️ The single most dangerous mistake in this project

> **Terrain must be ADDED. Never subtracted.**

If someone computes `DTM − AGL` instead, you get `200 − 18 = 182`. That number looks
completely normal. Nothing crashes. The 3D scene renders beautifully. Every building is
inside-out and **no automatic check catches it.**

Your project's own documentation says this has already bitten the team **twice**.

This is why the code we wrote refuses to save a file when it detects this. More on that
in Part 6.

---

## Part 3 — What existed before today

Ritvik's half was **done and frozen** (release `v0.1.0`). Your half was **a document
describing what someone should build.** Zero lines of code.

We checked:

```
grep -rn "dsm|DSM|calibrat" --include="*.py" .
→ only comments and docstrings. No implementation anywhere.
```

So the starting point was: an 8-step build order in a Markdown file, and nothing else.

---

## Part 4 — What we built

### 4.1 The four new code files

```
heightmap/
  dsm/
    terrain.py     ← works out the ground height (the "200")
    calibrate.py   ← fixes the model being slightly off + scores accuracy
    convert.py     ← does the "+", writes the output files
    __init__.py    ← ties the package together
  to_dsm.py        ← the command you actually type
```

Plus:

```
scripts/make_geotiff.py   ← turns a plain image into a proper map-aware file
tests/test_dsm.py         ← 12 tests proving all of it works
SPEC_DSM.md               ← the engineering contract
data/GAMUS_geotiff/       ← 6 ready-to-use GeoTIFFs for Ritvik
```

### 4.2 `terrain.py` — where the ground height comes from

Three ways to get the "200", ranked by how much you should trust them:

| Method | How it works | When to use | Accuracy |
|---|---|---|---|
| **GCP fit** | You surveyed a few real points; fit a tilted plane through them | You have survey data | ±0.5 m |
| **DEM opening** | Take an elevation file, strip the buildings off it | Big scenes, hills | ±3–8 m |
| **Constant plane** | One number for the whole scene | A single tile | ±1–2 m |

**Why "constant plane" is respectable, not lazy:** one tile is 1024×1024 pixels at
0.33 m/pixel = **338 × 338 metres**. Across 338 m of a city the ground rises or falls by
maybe one or two metres. That is *smaller than the model's own error bar* (1.585 m). So
assuming flat costs you less than it sounds like it should. It stops being defensible
past ~1 km, or anywhere with real relief.

### 4.3 `calibrate.py` — fixing the 2-metre drift

Remember the New York result: shape perfect, whole scene 2.02 m too low.

**The trick:** roads and bare ground are **0 metres tall by definition.** So if the model
says a road is 2 m tall, that 2 m *is* the model's bias. Measure it, subtract it, done.

This needs **no external data at all** — the model produces its own land-cover map, so it
tells you which pixels are road. Free calibration.

**Critical detail: fit on ground, never on buildings.** The building heights are the half
that is already right. Fitting there spends your correction on the good half and leaves
the offset in place.

It also computes **RMSE, MAE, correlation and signed bias** — the four numbers the
problem statement asks for (requirement 13), reported separately for ground and buildings.

> **Why "signed" bias matters:** two models can both score 4 m error. One is randomly off
> by ±4 m; the other is consistently 4 m short. Only the sign tells them apart — and a
> consistent offset is the fixable kind.

### 4.4 `convert.py` — the actual conversion

Does `DSM = DTM + AGL`, checks it, and writes the files. It supports **two modes**, and
picks automatically:

| Your input | Mode | Output | Why |
|---|---|---|---|
| A file with location info (GeoTIFF) | **absolute** | Real metres, e.g. 216–255 m | You know where you are, so you can know the ground height |
| A plain PNG/JPG | **relative** | 0.00 to 1.00 | No location = no way to look up ground height |

**Relative mode is not a failure.** The problem statement *specifically asks* for this
behaviour on plain images (requirement 5). Writing fake metres would be worse than
writing an honest relative surface. It still renders in 3D perfectly.

### 4.5 `to_dsm.py` — the command

This is what you type. It prints plain-English explanations of what it is doing and warns
you about anything questionable.

---

## Part 5 — How to run it (copy-paste)

### Step 1 — Make a map-aware image file

Your data (GAMUS) is `.h5` files: raw pixel grids with **no coordinates at all.** We
proved it:

```
keys inside file : ['image']
image: shape=(1024, 1024, 3) dtype=uint8
file-level attrs : {}   ← EMPTY: no CRS, no transform, no coords
```

So we convert:

```bash
python scripts/make_geotiff.py \
       --image data/GAMUS/images/val/DC_02_26_RGB.h5 \
       --lat 38.9072 --lon -77.0369 --gsd 0.33 \
       --out data/GAMUS_geotiff/DC_02_26.tif
```

**Already done for all 6 tiles** — they're committed in `data/GAMUS_geotiff/`.

### Step 2 — Inference (Ritvik's model)

```bash
python -m heightmap.predict --ckpt a7_vitl_v0.1.0.pt \
       --image data/GAMUS_geotiff/DC_02_26.tif \
       --out outputs/pred
```

Note: **no `--gsd` flag needed.** The GeoTIFF carries it. Verified working.

### Step 3 — Your stage

```bash
python -m heightmap.to_dsm --agl outputs/pred/DC_02_26.tif \
       --out outputs/dsm --base-elevation 10
```

`10` = Washington DC is ~10 m above sea level.

### Run the tests (great for a demo)

```bash
python tests/test_dsm.py
```

### Real output you'll see

```
mode  ABSOLUTE DSM (metres above sea level)
      terrain: flat plane at 216 m above sea level

wrote geo_agl_dsm.tif  +  sidecar  +  preview
      DSM  min=216.00m  p50=216.01m  max=248.22m
      terrain  216.00 - 216.00 m  (the DSM sits on top of this, never below it)
```

**Read that:** lowest point is 216 m — that's bare ground, correct. Highest is 248.22 m,
which is 216 + a 32 m building. The arithmetic is visibly right.

---

## Part 6 — The three traps (your best talking points)

Every one of these produces a file that **opens fine, renders fine, and is wrong.** No
error message. That is exactly what makes them dangerous, and defending against them is
the most impressive part of your work.

### Trap 1 — Subtracting terrain

Covered above. Our guard:

```python
assert_dsm_above_dtm(dsm, dtm, valid, agl=agl)
```

It runs on **every single write**. If the terrain was subtracted, it refuses to save.

**Subtle engineering detail worth mentioning to a judge:** the tolerance is 0.5 m, not
zero. After calibration, ground pixels legitimately sit a couple of *centimetres*
negative — that's the calibration working. A check strict enough to reject that would
fire constantly, get switched off, and take the real guard with it. So it allows
centimetres and catches metres.

### Trap 2 — Free elevation data is NOT the ground

This is the one that would most likely have shipped broken.

You can freely download **SRTM** or **Copernicus GLO-30** — global elevation data, 30 m
resolution. Everyone assumes it's bare ground.

**It isn't. It's a surface model.** ESA's own description says Copernicus includes
buildings, infrastructure and vegetation. **Over a city it already sits at rooftop
level.** So:

```
WRONG:  SRTM + AGL  =  rooftop + 18 m  =  a 36 m building
        Every building in your city, doubled.
```

Nothing errors. The render looks great.

**Our fix:** the `--dem` option *never* uses an elevation file directly. It first runs
`dem_to_dtm()`, a **morphological opening** — take the minimum over a sliding window,
then the maximum. Anything narrower than the window and standing above its surroundings
(a building) gets erased. Anything broader (a hill) survives.

**We proved it works.** A test DEM read **221.93 m** over the buildings; the filter
recovered **~202 m** of real bare earth.

⚠️ The window must be **wider than the widest building**, or the middle of a big flat
roof survives and becomes permanent fake terrain. Default 120 m. The tool prints how much
height it removed — if that's ~0 over a city, your window is too narrow.

### Trap 3 — Vertical datums (a silent 30–70 m error over India)

"Height above sea level" is **not one number.** There are two families:

| Family | Zero means | Examples |
|---|---|---|
| **Ellipsoidal** | A smooth mathematical model of Earth | Raw GPS, WGS84 |
| **Orthometric** | The geoid — real mean sea level, which is lumpy | SRTM (EGM96), Copernicus (EGM2008) |

The gap between them runs from about **−100 m to +85 m** worldwide. **Over India it's
roughly −30 m to −70 m.**

Mix one with the other and every elevation is tens of metres wrong — no error, no
warning, perfectly normal-looking render.

**Our fix:** every file we write states its datum, in the sidecar *and* as a file tag. The
tool warns when a DEM declares none.

---

## Part 7 — What the output looks like

Three files per scene:

| File | What it is |
|---|---|
| `X_dsm.tif` | **The deliverable.** Real metres, float32. Aakarsh's viewer reads this |
| `X_dsm.json` | The receipt — every assumption stated |
| `X_dsm_preview.png` | A picture for slides |

> ⚠️ **Never read numbers off the preview PNG.** It's 8-bit and stretched for human eyes.
> On a nearly-flat scene it turns 1 metre of noise into what looks like a mountain range.
> The heights in it are fiction. Our code physically refuses to accept a PNG as input.

### The receipt — what a judge actually reads

```json
{
  "quantity": "DSM",
  "units": "metres",
  "vertical_datum": "EGM2008 (orthometric, approx. mean sea level)",
  "georeferenced": true,
  "crs": "EPSG:32618",
  "pixel_spacing_m": 0.33,
  "dtm_source": "constant_plane",
  "assumptions": ["terrain treated as flat across the tile at 216 m"]
}
```

**That `assumptions` array is the most important thing in the file.** You *did* assume
the ground is flat. Saying so out loud is engineering. Hiding it is a bug waiting for
someone else to find.

> A judge who sees you declare your own shortcuts will trust the rest of your work
> **more**, not less.

---

## Part 8 — What is tested

`python tests/test_dsm.py` → 12 tests, no data or internet needed.

| Test | What it proves |
|---|---|
| `addition` | 18 m building on 200 m ground = 218 m |
| `subtraction_is_caught` | The catastrophic sign error is detected |
| `calibration_noise_is_not_caught` | Guard doesn't fire on legitimate cm-scale noise |
| `rdsm_survives_an_outlier` | One 390 m freak value doesn't flatten the scene |
| `nodata_never_reaches_a_statistic` | `-9999` never averaged in as a height |
| `dem_opening_strips_buildings` | Buildings removed, real hills preserved |
| `dem_window_too_narrow` | The documented failure mode is reproducible |
| `calibration_recovers_a_known_offset` | Recovers the exact 2.02 m NYC bias |
| `calibration_uses_road_as_ground_too` | Roads counted as ground |
| `few_samples_refuse_a_free_scale` | Won't overfit on a handful of points |
| `gcp_plane_recovers_a_tilt` | Survey-point fitting is exact |
| `metrics_report_signed_bias` | RMSE/MAE/r/bias split by land cover |

All 6 test files in the repo pass.

---

## Part 9 — The environment fixes (invisible but essential)

Your laptop **could not run the pipeline at all** when we started. Two separate problems:

1. **`omegaconf` was missing** — `predict.py` couldn't even be imported.
2. **numpy version clash** — you have numpy 2.4.6, but `h5py`, `pandas`, `numexpr` and
   `bottleneck` were compiled against numpy 1.x. Every import crashed with
   `numpy.dtype size changed`.

Fixed by installing `omegaconf` and upgrading the four broken packages. We deliberately
did **not** downgrade numpy or touch your Anaconda base environment beyond that — the
night before a hackathon is the wrong time to blow up a working Python install.

We also generated the missing test fixtures (`scripts/make_synthetic.py`), which took the
suite from 3 failing files to **0**.

---

## Part 10 — Honest limitations

State these before a judge finds them. Being upfront is worth more than the marks you'd
lose.

| Limitation | The honest position |
|---|---|
| **Coordinates are declared, not surveyed** | GAMUS doesn't record where each tile sits. We used the DC city centre, stepped per tile. Files say `georeferencing: "declared by operator, not surveyed"`. Fine for placing a scene and fetching a DEM; **not** a positional measurement |
| **`DSM = DTM + AGL` never validated on real data** | Nobody in the project has checked it against ground truth. US3D (DFC2019) is the dataset that would allow it — 15 GB download, not done |
| **No DEM auto-download** | You must supply a DEM file manually. Fetching needs network |
| **Never tested on Indian imagery** | No public Indian dataset with ground truth exists. Any claim about Indian accuracy is unverifiable until someone flies a drone |
| **Realistic accuracy** | 2–4 m error in cities. Published work on similar cities reports 2.2–7.0 m, so **3–5 m for India** is what you can defend |
| **Hilly terrain fails** | The model returns ~0 everywhere on a bare mountain. That is *correct* — AGL is height above local ground, and a mountain **is** the ground. Mountains need the terrain path, not the model path |

---

## Part 11 — Likely judge questions, with answers

**"Where does the terrain come from?"**
> Three options depending on what you have: surveyed control points, an elevation file
> with the buildings filtered out, or a flat plane. We record which one was used in every
> output file.

**"Why is a flat plane acceptable?"**
> One tile covers 338 × 338 metres. Across that distance urban ground varies by one or
> two metres — less than the model's own 1.585 m error bar. And we state the assumption
> in the file rather than hiding it.

**"Why not just use SRTM as your terrain?"**
> Because SRTM is a *surface* model, not bare earth. Over a city it sits at rooftop
> level, so adding our heights to it would double every building. We run a morphological
> opening filter first to strip the buildings out.

**"How do you know you didn't subtract instead of add?"**
> There's an assertion on every write that verifies `DSM − DTM` equals the AGL we put in.
> If it doesn't, the file isn't saved. There's a test that deliberately does the
> subtraction and confirms it's caught.

**"What's your accuracy?"**
> The height model scores 1.585 m building MAE with 0.901 correlation on a city it never
> trained on. Our stage adds terrain and removes a measured 2.02 m offset. We report
> RMSE, MAE, correlation and signed bias, split by ground and building — a pooled number
> flatters you, because a third of pixels are ground at exactly 0 m.

**"Is this tested on Indian data?"**
> No, and we say so. No public Indian dataset with ground truth exists. We'd expect 3–5 m
> based on published results for comparable cities. Verifying it honestly needs a drone
> survey.

---

## Part 12 — Glossary

| Term | Meaning |
|---|---|
| **AGL** | Above Ground Level. How tall a thing is above the ground under it |
| **DTM** | The bare ground with everything removed |
| **DSM** | The elevation of the top of everything |
| **rDSM** | A relative surface, 0 to 1, when real metres aren't possible |
| **GSD** | Ground Sample Distance — how many metres one pixel covers |
| **GeoTIFF** | An image file that also stores *where on Earth* it is |
| **CRS** | Coordinate Reference System — which map grid the coordinates use |
| **UTM** | A map projection where one unit = one metre |
| **EPSG:32618** | UTM zone 18 North — the grid Washington DC sits in |
| **DEM** | Any elevation file. Might be a DTM or a DSM — **you must check which** |
| **SRTM / Copernicus GLO-30** | Free global elevation data. Both are DSMs, not DTMs |
| **Geoid** | The lumpy "real" sea level surface |
| **EGM2008 / EGM96** | Two standard geoid models. They differ slightly |
| **nodata** | A fake value (`-9999`) meaning "no measurement here" |
| **MAE** | Mean Absolute Error — average size of the mistake |
| **RMSE** | Like MAE but punishes big mistakes harder |
| **Signed bias** | Average error *with direction* — are we too high or too low? |
| **Morphological opening** | Min then max over a window. Erases small tall things |
| **LiDAR** | Laser scanning. The expensive, accurate way to measure height |

---

## Part 13 — What was committed

| Commit | What |
|---|---|
| `8002bcd` | The AGL → DSM stage: 4 modules, CLI, 12 tests, `SPEC_DSM.md` |
| `cb385ac` | `make_geotiff.py` — wrap a plain image in a real GeoTIFF |
| `32f5e1c` | Merge of Vineeth's TDA module from `main` |
| `a96c953` | 6 GAMUS tiles + their GeoTIFF conversions, for Ritvik |

All authored by **Divyanshu Sharma**, on `export-pipeline`, merged to `main`.

**Requirements covered:** PS 2, 3, 4, 5, 6 and 13.
