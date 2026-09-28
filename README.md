# DepthWizard

**One overhead image in. A metric 3D scene out.**

Team DepthX · Smart India Hackathon 2026 · Problem statement **SIH26175**

**Live:** [sih-2026-depth-x.vercel.app](https://sih-2026-depth-x.vercel.app)

DepthWizard estimates the height of every pixel, in metres, from a single nadir satellite
or aerial image. When the image is georeferenced it places those heights on real terrain to
form a Digital Surface Model (DSM), writes a proper GeoTIFF, and renders the result in the
browser as a scene you can orbit, fly and walk through.

| | |
|---|---|
| Height error, 10 Swiss scenes with airborne LiDAR, never seen in training | **MAE 1.74 m · RMSE 3.01 m · r 0.913** |
| Urban / sparse / hilly / forested (MAE) | 1.59 m / 0.31 m / 1.74 m / 3.40 m |
| Buildings in a city held out of training (New York) | MAE 1.59 m · r 0.901 |
| Model inference, 1024 px scene, NVIDIA L4 | **1.1 s** |
| Deployment | Vercel (web) + Modal (GPU), fully standalone |

All accuracy numbers are for the shipped checkpoint (`a7_vitl`, release v0.1.0), on every
valid pixel of data the model never trained on. The Swiss figures were measured through the
live deployment and reproduce with one command (see
[Reproducing the numbers](#reproducing-the-numbers)).

---

## Contents

1. [How it works](#how-it-works)
2. [Criterion 1: DSM estimation, accuracy and validation](#criterion-1-dsm-estimation-accuracy-and-validation-50)
3. [Criterion 2: Visualization, rendering quality and experience](#criterion-2-visualization-rendering-quality-and-experience-50)
4. [Try it](#try-it)
5. [Run it yourself](#run-it-yourself)
6. [Repository layout](#repository-layout)
7. [Documentation](#documentation)
8. [Data and credits](#data-and-credits)

---

## How it works

```
one nadir RGB image (GeoTIFF, PNG or JPEG)
  │
  ▼  HeightNet: DINOv2 ViT-L encoder + DPT neck + adaptive-bins head, conditioned on pixel size
height above local ground (AGL), metres, float32
  │
  ▼  georeferenced input only: Copernicus GLO-30 terrain, reduced to a bare-earth DTM
DSM = DTM + AGL, GeoTIFF with CRS, nodata and a metadata sidecar
  │
  ▼  browser: Three.js mesh at up to 1024 x 1024 segments, textured with the source image
navigable 3D scene
```

**The model.** The encoder and neck come from Depth Anything V2 Large. Its original depth
head is **replaced, not calibrated**: that head was trained with a loss that ignores scale
and shift, so metric height is not recoverable from it by rescaling. We measured this: stock
Depth Anything V2 with the best global scale and shift fit reaches only r 0.405 on our data. The new
head predicts height through **adaptive bins** (a pixel on a roof edge is roof or ground,
never the blurred average) and is **conditioned on ground sample distance** through FiLM,
which is what lets it tell a 40 px shed at 0.3 m/px from a 40 px warehouse at 2 m/px.
Training used GAMUS: 8,724 tiles over Washington DC, New York and Philadelphia at 0.33 m/px,
with scale augmentation covering 0.165 to 0.65 m/px. Details and the ablation ladder are in
[`MODEL_CARD.md`](MODEL_CARD.md).

**Terrain.** A single image carries height above ground, not elevation. For a georeferenced
input the pipeline fetches the Copernicus GLO-30 elevation model over the same footprint,
reduces it to bare earth with a 120 m morphological opening (the raw product is itself a
surface model that sits on rooftops), and adds the predicted heights. Contract and datum
handling: [`SPEC_DSM.md`](SPEC_DSM.md).

**Serving.** The model runs on an NVIDIA L4 on Modal that exists only while someone is using
the site. Loading the page never starts a GPU; the Warm up button or an upload does, and the
GPU switches itself off about 15 s after the last page closes.

---

## Criterion 1: DSM estimation, accuracy and validation (50%)

### How we measured

| Reference set | What it covers | How independent |
|---|---|---|
| **swisstopo**, 10 scenes, 338 x 338 m each | Alpine and lakeside towns, 24 to 115 m of relief, 30 to 57% built | **Fully unseen**: a different country, sensor and vegetation. Surface and terrain from airborne LiDAR (swissSURFACE3D, swissALTI3D). Scored through the **live deployment** by [`scripts/evaluate_live.py`](scripts/evaluate_live.py); raw output in [`results/validation.json`](results/validation.json). |
| **GAMUS test split**, 800 tiles | US urban, suburban and wooded scenes with per-pixel land-cover labels | Tiles never trained on, from cities that were. One centre 518 x 518 crop per tile, scored offline on the training cluster; summary in [`results/offline_eval.json`](results/offline_eval.json). The website's tiled path scores about 5% better. |
| **New York, held out** | A whole city | The same recipe retrained with New York removed, then scored on New York. |

Metrics: **MAE** (mean absolute error), **RMSE** (root-mean-square error, which weights large
misses), **bias** (signed mean error) and **Pearson r** (structure, independent of scale).
The GAMUS validation split and the in-domain score (building MAE 1.549 m) are not quoted
here, because that split was used to choose the checkpoint.

### Accuracy by landscape

Height above ground (AGL), the model's own output:

| Landscape | Data | MAE | RMSE | Bias | r |
|---|---|---|---|---|---|
| **Urban** | GAMUS test, 320 tiles with 15% or more buildings | **1.59 m** | 2.86 m | −0.06 m | **0.930** |
| **Sparse** | GAMUS test, 103 tiles under 5% buildings and under 10% trees | **0.31 m** | 1.05 m | +0.02 m | 0.737 |
| **Hilly** | swisstopo, 10 scenes, all pixels, live model | **1.74 m** | 3.01 m | −0.51 m | **0.913** |
| **Forested** | GAMUS test, 293 tiles with 30% or more trees | **3.40 m** | 5.51 m | +1.20 m | 0.846 |
| Mixed | GAMUS test, the remaining 84 tiles | 0.89 m | 1.98 m | +0.08 m | 0.890 |

Whole-landscape MAE spans 0.31 to 3.40 m. Sparse scenes are almost flat, so there is little
height for r to track; their MAE is the meaningful figure. **Forested is the weakest
landscape**: canopy height is irregular and hard to read from a single image.

### Generalisation to an unseen city

| New York, held out of training | MAE | RMSE | Bias | r |
|---|---|---|---|---|
| Buildings | 1.59 m | 2.63 m | −0.36 m | 0.901 |
| All pixels | 2.91 m | 5.63 m | −2.12 m | 0.728 |

Buildings transfer almost unchanged. Across all pixels a systematic offset appears on the new
city (−2.1 m), which is why correlation drops more than building error does. One checkpoint
throughout: best building MAE, epoch 37.

### Inside the Swiss scenes

The live Swiss run, split two more ways. Built fraction and relief per scene:

| Group | Scenes | MAE | RMSE | Bias | r |
|---|---|---|---|---|---|
| Dense towns, 55% or more built | Baden, Neuchâtel, Schaffhausen, Vevey | 2.19 m | 3.51 m | −0.92 m | 0.904 |
| Sparse, 36% or less built | Sierre, Spiez, Weggis, Zug | 1.22 m | 2.29 m | −0.11 m | 0.914 |
| Steepest, 80 to 115 m relief | Baden, Spiez, St. Moritz, Weggis | 1.77 m | 3.09 m | −0.74 m | 0.895 |

By land cover (swisstopo has no class labels, so this split is approximate: above 2.5 m and
green by an excess-green index is vegetation, above 2.5 m otherwise is a building):

| Land cover | MAE | RMSE | Bias | r |
|---|---|---|---|---|
| Buildings | 3.10 m | 4.17 m | −1.65 m | 0.814 |
| Trees and vegetation | 3.20 m | 4.37 m | −2.04 m | 0.773 |
| Open ground | 0.60 m | 1.34 m | +0.53 m | 0.342 |

### Surface model (DSM) against LiDAR

swisstopo, 10 scenes, DSM = terrain + predicted AGL, scored against the LiDAR surface:

| Terrain under the heights | MAE | RMSE | Bias | r (median per scene) |
|---|---|---|---|---|
| **As deployed**: Copernicus GLO-30, free, 30 m | **3.88 m** | **5.30 m** | +1.47 m | 0.947 |
| LiDAR terrain (the ceiling: model error only) | 1.78 m | 3.04 m | −0.48 m | 0.973 |
| Copernicus terrain on its own | 3.64 m | 4.94 m | +1.95 m | |

About 2 m of the deployed DSM error comes from the free 30 m terrain model, not from the
height model; supplying a better DTM (`--dem`) recovers it. Correlation is the median per
scene: pooled across scenes a DSM correlates at 0.9999, but that only says a town at 1,800 m
sits higher than one at 400 m, so we do not quote it.

### Stability across scenes

| Scene | Relief | Built | MAE | RMSE | r |
|---|---|---|---|---|---|
| St. Moritz | 115 m | 49% | 2.01 m | 3.34 m | 0.905 |
| Weggis | 98 m | 32% | 1.23 m | 2.40 m | 0.917 |
| Baden | 96 m | 57% | 2.85 m | 4.28 m | 0.878 |
| Spiez | 87 m | 30% | 1.00 m | 1.72 m | 0.913 |
| Montreux | 46 m | 41% | 1.79 m | 3.06 m | 0.910 |
| Neuchâtel | 36 m | 57% | 2.04 m | 3.19 m | 0.936 |
| Vevey | 35 m | 55% | 1.86 m | 3.10 m | 0.927 |
| Schaffhausen | 32 m | 57% | 2.01 m | 3.36 m | 0.903 |
| Sierre | 31 m | 34% | 1.39 m | 2.54 m | 0.892 |
| Zug | 24 m | 36% | 1.28 m | 2.39 m | 0.946 |

Correlation stays between 0.878 and 0.946 from flat lakeside towns to a 115 m alpine slope.
Error follows how built-up a scene is more than how steep it is: the lowest error of all is on
an 87 m hillside (Spiez), while dense towns average 2.19 m against 1.22 m for sparse scenes.

### Known limitations

- **Forested scenes are the weakest** (3.40 m MAE). Canopy is irregular, and in Switzerland
  vegetation is underestimated by about 2 m.
- **Heights are softer than truth.** Building edges come out blurred over roughly 2 m, and
  tall buildings read low (−1.65 m bias on Swiss buildings). The viewer sharpens edges for
  display only; the GeoTIFF keeps the model's values.
- **Trained pixel sizes are 0.165 to 0.65 m/px.** Coarser imagery (for example 1 m) is outside
  that range; `scripts/predict_external.py` resamples before inference.
- **The deployed DSM is limited by free 30 m terrain**, which alone is off by 3.6 m MAE.
  Supplying a better DTM (`--dem`) removes that error.
- **No Indian LiDAR was available to validate against.** Indian scenes on the site are
  qualitative.

---

## Criterion 2: Visualization, rendering quality and experience (50%)

**Projection accuracy**
- The texture is resampled onto the exact grid of the height raster, so every image pixel sits
  on its own height sample. A built-in **alignment check** paints the heights over the terrain
  so any offset is visible at once.
- Ground scale comes from the GeoTIFF itself (pixel size, CRS); terrain scenes render at true
  1:1 vertical scale.
- Scenes are rendered relative to their lowest point, so a DSM at 5,500 m (North Sikkim)
  frames and navigates exactly like a flat city.
- Parity check: the cloud GPU and the reference pipeline agree to **0.2 cm mean, 2.2 cm
  worst pixel**, correlation 1.000000.

**Visual fidelity**
- Mesh up to 1024 x 1024 segments (1.05 M vertices) at full raster resolution.
- Faceted shading for crisp walls, facade detail on detected buildings, roof-edge highlights,
  instanced trees that replace canopy mounds, directional sun with shadows.
- Height colour ramp and wireframe views for inspection.

**Navigability**
- Orbit, Fly and Walk modes. Collision is tested against the **rendered** surface, so the
  camera cannot fall through terrain, and the HUD reports height above ground live.
- A minimap of the source image shows the camera position and heading. It opens a comparison
  view with photo, height map and swipe modes, zoom, pan, a scale bar and the height under
  the cursor.

**Interface**
- Five precomputed sample scenes open in seconds with no GPU, so anyone can evaluate the
  viewer instantly.
- Georeferencing is detected in the browser before upload; terrain is added automatically
  when it applies and skipped when it does not.
- Separate live timers for upload, **GPU start-up**, **model inference**, terrain and
  download, so a cold start is never mistaken for a slow model.

**Stability**
- 15 of 15 runs clean across cold and warm starts with results downloaded in parallel;
  downloads retry, and a failure stops every timer and names the file.
- A 2D relief fallback when WebGL is unavailable, and error boundaries around the viewer.

**Standalone deployment**
- Web app on **Vercel**, model on **Modal** (NVIDIA L4). No local installation, no tunnel.
- GPU start-up 9 to 44 s, once, then about 1.1 s of model time per 1024 px scene.
- Cost guards: at most one GPU, 20 reconstructions per IP per hour, 400 per day; the GPU
  sleeps when nobody is using it.
- A private run log at `/admin` records every reconstruction with its image, timings and
  approximate location.

---

## Try it

1. Open the site and pick a **sample scene**. It opens in seconds with no GPU.
2. Press **Warm up GPU**. A sleeping GPU takes 20 to 30 seconds to start.
3. Drop a top-down **GeoTIFF, PNG or JPEG** and press **Generate 3D scene**.
4. Explore: drag to orbit, scroll to zoom, or switch to Fly or Walk. Layers are on the top bar,
   and the source image in the corner opens the comparison view.

---

## Run it yourself

### Website

```bash
cd terrain3d
npm install
npm run dev            # http://localhost:3000, uses the Modal service by default
```

`?api=<url>` points a tab at another backend, for example a local `serve/app.py`.

### Model service on Modal

```bash
pip install modal
modal token set ...                              # your Modal account
modal deploy serve/modal_app.py                  # prints the permanent URL
```

The checkpoint (`a7_vitl_v0.1.0.pt`, 1.25 GB, GitHub release `v0.1.0`) goes into the
`depthx-weights` volume; `serve/fetch_weights_modal.py` copies it from the release inside
Modal and checks its SHA-256.

### Command-line pipeline (any CUDA or Apple Silicon machine)

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --ckpt a7_vitl_v0.1.0.pt --image scene.tif --out run1/ --auto-dem
```

Writes `run1/bundle/` (heightmap, texture, metadata, the viewer's input) and `run1/raw/`
(the full output contract of [`SPEC.md`](SPEC.md)).

### Reproducing the numbers

```bash
pip install modal numpy pillow h5py
python scripts/evaluate_live.py --out results/validation.json   # scores the live model on swisstopo
python scripts/make_site_metrics.py                             # refreshes the website's metrics card
```

### Training

```bash
python -m heightmap.train --config=configs/ablations/a7_vitl.yaml data.root=/path/to/gamus
```

---

## Repository layout

```
heightmap/            the model: data loading, network, losses, tiled inference, export
  dsm/                AGL to DSM: terrain, calibration, absolute and relative outputs
configs/              base config and the ablation ladder (A1 to A7, holdout)
scripts/              pipeline, evaluation, data download, sample packaging
serve/
  modal_app.py        the deployed service: web tier + L4 GPU class on Modal
  app.py              the same HTTP contract as a plain FastAPI server
terrain3d/            the website (Next.js + Three.js)
  components/         viewer, controls, comparison view, homepage, metrics
  lib/                mesh building, scene analysis, loaders, GPU lifecycle
  public/samples/     the five precomputed scenes
results/              measured accuracy: live Swiss run, offline GAMUS and New York scores
data/                 swisstopo reference scenes, a GAMUS validation sample
tda/                  topology-based refinement and its evaluation
tests/                unit tests for inference, metrics and the DSM stage
hpc/                  training job scripts for the PBS cluster
final_test_files/     the five demo inputs behind the sample scenes
```

---

## Documentation

| File | What it covers |
|---|---|
| [`MODEL_CARD.md`](MODEL_CARD.md) | Architecture, training, results, what did not work, limitations |
| [`PIPELINE.md`](PIPELINE.md) | How the pipeline runs end to end |
| [`SPEC.md`](SPEC.md) | Output contract of the height stage |
| [`SPEC_DSM.md`](SPEC_DSM.md) | Output contract of the AGL to DSM stage, datums |
| [`data/swisstopo/README.md`](data/swisstopo/README.md) | The Swiss reference scenes and their licence |

---

## Data and credits

- **GAMUS**: Xiong et al., *GAMUS: A Geometry-aware Multi-modal Semantic Segmentation
  Benchmark for Remote Sensing Data*, arXiv:2305.14914.
- **Depth Anything V2**: Yang et al., 2024 (encoder and neck).
- **swisstopo**: Federal Office of Topography swisstopo, SWISSIMAGE, swissALTI3D,
  swissSURFACE3D. https://www.swisstopo.admin.ch
- **Copernicus GLO-30 DEM**: produced using Copernicus WorldDEM-30, © DLR e.V. 2010-2014 and
  © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union
  and ESA.

Deployment fix 1.
