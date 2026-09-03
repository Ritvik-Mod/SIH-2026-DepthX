# DepthWizard — what is built, and how the pipeline actually runs

**For Aakarsh (deployment + Three.js), Divyanshu (calibration), Vineeth (TDA).**

This documents what **exists and works today**, not what is planned. Anything not built
is in `PS_COVERAGE.md`, not here. Last updated 4 Sept 2026.

---

## 1 · The one-line version

```
RGB image  →  HeightNet (DINOv2 ViT-L + DPT + metric bins head)  →  AGL in metres
           →  export scripts  →  heightmap.tif + texture.png + metadata.json  →  Three.js
```

**AGL = height above LOCAL GROUND, in metres. Not elevation above sea level.**
`DSM = DTM + AGL`. Terrain is **added**. Subtracting produces a scene that looks
plausible and is completely wrong, and nothing downstream catches it.

---

## 2 · What was actually built

### The model
- **HeightNet**: DINOv2 ViT-L/14 encoder + DPT neck, both inherited from Depth Anything
  V2 Large. The DAv2 **depth head was discarded and replaced** with:
  - adaptive-bins height head (128 bins, `h_max` 250 m)
  - FiLM conditioning on `log(GSD)`
  - auxiliary 7-class semantic head
- **335.4M parameters.**
- Losses: SILog + L1 + multi-scale gradient + bin CE + chamfer + semantic CE.
  Shadow-consistency loss is implemented but **off** (measured negative, see §8).

### Trained and measured
- Ablation ladder **A0 → A7** complete on GAMUS (8,724 tiles, 3 US cities, 0.33 m GSD).
- **A7 (ViT-L) is the shipped model.** In-domain building MAE **1.549 m**.
- **Held-out city (NYC, never trained on): building MAE 1.585 m, r 0.901.**
- Two independent unseen splits agree to 2.5% (test 1.600 m, val 1.560 m).
- 22 bugs found and fixed, documented in `HANDOFF.md` §7 and `HANDOFF_SESSION2.md` §2.
- 5 test modules pass; `scripts/sanity.py` is 16/16 on real GAMUS.

### Infrastructure
- PBS job scripts for BIT Mesra HPCF (`hpc/`), atomic checkpointing, resume-safe.
- Tiled full-scene inference with coarse-guided levelling and ramp blending.
- Export scripts producing the Three.js bundle (this document, §5).
- Fixed-ratio multi-domain batch mixing and block-level fine-tuning surgery
  (`heightmap/data/mixing.py`, `net.py`) — built and tested, not yet used in a run.

---

## 3 · Model weights — how to get them

**GitHub Release `v0.1.0`** on `Ritvik-Mod/SIH-2026-DepthX`:

| | |
|---|---|
| asset | `a7_vitl_v0.1.0.pt` |
| size | **1,341,893,322 bytes (1.25 GB)** |
| sha256 | `27c4098c4689ff7866fa09276637107c79cee9bf1ca7b1dd9d5d6cb9d4dbe1df` |
| contents | weights only — optimiser state stripped, EMA merged |
| URL | `https://github.com/Ritvik-Mod/SIH-2026-DepthX/releases/download/v0.1.0/a7_vitl_v0.1.0.pt` |

```bash
gh release download v0.1.0 --repo Ritvik-Mod/SIH-2026-DepthX --pattern '*.pt'
```

The EMA weights are merged in, because **the reported metrics are EMA numbers** — the
released model and the released metrics therefore agree. This checkpoint **cannot be
resumed from** (no optimiser state); that is deliberate. Training continues from
`outputs/a7_vitl/best.pt` on the cluster (5.0 GB).

---

## 4 · Stage 1 — image in, AGL out

### Accepted input

| property | value |
|---|---|
| formats | `.png` `.jpg` `.jpeg` `.tif` `.tiff` `.webp` |
| channels | RGB (greyscale is expanded to 3; alpha is dropped) |
| dtype | uint8 |
| size | any — the scene is tiled internally |
| **GSD** | **must be known or declared, in metres/pixel** |

**GSD is the parameter that matters most.** The model is conditioned on `log(GSD)`
through FiLM and was trained over **0.165–0.65 m/px**. The image is resampled so its
effective GSD lands inside that band before inference. Declare it wrong by 2× and every
output metre is wrong by 2×.

> Reference points: a car is ~4.5 m long, a road lane ~3.5 m wide.
> `GSD = metres ÷ pixels`.

Out-of-band imagery is not refused — it warns and extrapolates. A Himalayan scene at
~20–40 m/px returned p50 0.3 m / max 1.1 m, i.e. nothing, which is the *correct* AGL
answer over bare terrain and a useless terrain answer. See `PS_COVERAGE.md`.

### What happens inside

1. **Resample** to the target GSD (default 0.33 m/px), long side capped at 2048 px.
2. **Coarse pass** over the whole scene downscaled to one 518² tile, conditioned on the
   coarser GSD. This gives a global height reference so tiles do not drift apart.
3. **Tiled pass**: 518×518 windows at **25% overlap**, batched, each blended with a ramp
   window. Overlap averaging measurably *improves* accuracy — building MAE 1.588 → 1.502
   versus a single crop — because it is an ensemble, not a smoother.
4. **Levelling**: each tile is offset toward the coarse reference before blending.

518 is not arbitrary: patch size is 14, and 518 = 37 × 14. **Any model input must be a
multiple of 14** or it raises.

### Output of stage 1

`InferResult`, from `heightmap/infer.py`:

| field | type | meaning |
|---|---|---|
| `agl` | `(H, W)` float32 | **height above local ground, metres.** Clipped at ≥ 0 |
| `sigma` | `(H, W)` float32 or `None` | per-pixel 1σ uncertainty; `None` unless the uncertainty head is enabled (it is **off** in v0.1.0) |
| `overlap_disagreement` | `(H, W)` float32 or `None` | spread between overlapping tile predictions — a free confidence proxy |
| `meta` | dict | tile/overlap/GSD actually used |

`H, W` match the **resampled** image, not the original. The original dimensions are
recorded in `metadata.json`.

---

## 5 · Stage 2 — export, the files Aakarsh consumes

Produced by `scripts/export_for_threejs.py` (one scene) and
`scripts/export_batch_threejs.py` (many, model loaded once). Both live on `main`.

### Per-scene directory

```
<scene_name>/
  heightmap.tif           <- THE GEOMETRY. float32 GeoTIFF, REAL METRES
  texture.png             <- RGB, pixel-for-pixel aligned to heightmap.tif
  heightmap_preview.png   <- 8-bit grey, normalised. VIEWING ONLY, NOT DATA
  metadata.json           <- ranges, spacing, provenance, accuracy
  <scene>_agl.npy         <- the same float32 field as .npy, if easier to load
  <scene>_rgb.png         <- source RGB before texture matching
```

Plus `manifest.json` at the root of a batch run, listing every scene with its height
range and accuracy.

### `heightmap.tif` — the file the mesh is built from

| property | value |
|---|---|
| bands | **1** |
| band name | `agl_metres` |
| dtype | **float32** |
| units | **metres**, already real — do **not** rescale |
| nodata | **`-9999.0`** |
| compression | DEFLATE, internally tiled 256×256 |
| georeferencing | **absent** in v0.1.0 (`crs` / `transform` plumbed but passed `None`) |
| tags | `quantity=AGL`, plus the add-don't-subtract note |

> ⚠️ **Two export paths exist and they differ.** `heightmap/export.py` writes the
> **3-band** contract in `SPEC.md` (`agl_metres`, `sigma_metres`, `agl_normalised`).
> `scripts/export_for_threejs.py` writes **1 band**. **Every bundle Aakarsh has been
> given is 1-band.** Read band 1 and do not assume the others exist.

### `texture.png`
Matched to the heightmap's exact pixel dimensions. If the aspect ratios differ the RGB
is **centre-cropped first, then resized** — never stretched, because stretching would put
a roof beside its own footprint and look like a model error that isn't one.

### `heightmap_preview.png` — the trap
8-bit grey, stretched between the **1st and 99th percentile**. It is a viewing aid.
Using it as geometry silently rescales everything, and on a near-flat scene it amplifies
noise into fake mountains — that is exactly what happened on the Himalaya test, where
1.1 m of noise rendered as a mountain range.

### `metadata.json`

```json
{
  "quantity": "AGL",
  "quantity_note": "height above LOCAL GROUND, in metres. DSM = DTM + heightmap. ADD the terrain model, do not subtract.",
  "units": "metres",
  "min_height_m": 0.0, "max_height_m": 24.29, "mean_height_m": 3.38,
  "resolution": {"width": 1024, "height": 1024},
  "pixel_spacing_m": 0.33,
  "ground_extent_m": {"width": 337.92, "height": 337.92},
  "geotiff_dtype": "float32",
  "nodata": -9999.0, "nodata_pixels": 0,
  "georeferenced": false,
  "field_source": "model_prediction",
  "split": "train",
  "accuracy_vs_lidar": {"building": {"mae": 0.98, "r": 0.95}, "overall": {...}},
  "texture_match": "already matched"
}
```

**`field_source`** is the field to check before trusting anything:
`model_prediction` vs `ground_truth_lidar`. A GT heightmap and a predicted one are
byte-shaped identically — without this field a folder of LiDAR truth mistaken for model
output makes the model look perfect.

---

## 6 · Building the mesh — the numbers you need

```
mesh_width_metres  = width  × pixel_spacing_m
mesh_depth_metres  = height × pixel_spacing_m
vertex_height      = heightmap.tif band 1 value        (already metres, no scaling)
```

For a 1024² tile at 0.33 m/px: **337.92 × 337.92 m** of ground.

Vertical exaggeration is a *display* choice — apply it in the renderer, never bake it
into the file. And mask `-9999.0` **before** computing any min/max/mean: it is a
sentinel, not a height, and averaging it in shifts a mean by kilometres.

---

## 7 · Running it yourself

**Environment** (cluster): `~/envs/depthwizard`, Python 3.11, torch 2.6.0+cu124.
Local dev: `.venv`, Python 3.13, torch 2.13, Apple MPS.

```bash
source /apps/anaconda3/bin/activate deeplearning
source ~/envs/depthwizard/bin/activate
cd ~/SIH2026
export HF_HOME=~/SIH2026/data/hf_cache
export HF_HUB_OFFLINE=1
```

`HF_HOME` matters — without it the backbone re-downloads 1.27 GB, and compute nodes
have no internet.

**Arbitrary external imagery** (this is the entry point for the app):

```bash
python scripts/predict_external.py <folder-or-file> \
  --ckpt a7_vitl_v0.1.0.pt \
  --source-gsd 0.5 \
  --target-gsd 0.33 \
  --max-px 2048 \
  --device cuda \
  --out outputs/my_export
```

**GAMUS tiles, prediction and ground truth side by side:**

```bash
python scripts/export_batch_threejs.py --source pred --ckpt a7_vitl_v0.1.0.pt \
  --split test --from-json outputs/demo12_test/compare.json --out outputs/pred_export
python scripts/export_batch_threejs.py --source gt \
  --split test --from-json outputs/demo12_test/compare.json --out outputs/gt_export
```

`--source gt` packages the LiDAR truth through the **identical** path. That is the
control that separates a renderer bug from a model bug — if a scene looks wrong from the
GT bundle, the problem is not the model.

**Runtime:** ViT-L inference is ~8 img/s on one L40 for training-shaped batches; real
tiled inference is roughly 3× slower than that figure suggests. A 1024² tile is a few
seconds on GPU, a few minutes on CPU.

---

## 8 · Limitations that affect deployment

| limitation | detail |
|---|---|
| **AGL, not DSM** | terrain must be added downstream. The pipeline does not emit a DSM today |
| **No georeferencing** | `heightmap.tif` carries no CRS/transform in v0.1.0 |
| **GSD band 0.165–0.65 m/px** | outside it, heights are extrapolation. Cartosat-3 pan (0.25 m) is inside; 1 m Cartosat-2S MX is outside |
| **Urban only** | trained on 3 US cities. Bare terrain returns ~0 (correctly). Sparse and forested never measured as categories |
| **Soft edges** | sharpness ratio 0.084–0.133 against 1.0 for truth ≈ the truth blurred at σ ≈ 3 px ≈ 2 m |
| **Border ~130 px** | ~8% worse building MAE toward scene edges — less overlap coverage and shadows falling outside the frame. Prefer the interior |
| **Worst unseen tile** | building MAE **46.6 m**. Know this before you are asked |
| **No uncertainty band** | the uncertainty head is off in v0.1.0, so `sigma` is `None` |
| **Shadow loss is off** | measured negative: 0.4% at weight 0.1 over 38% of tiles, on a properly controlled comparison |
| **No Indian validation** | external runs are qualitative only — no ground truth, assumed GSD |

**Numbers safe to quote:** held-out city building MAE **1.585 m**, r **0.901**;
content-rich median on unseen test tiles **1.600 m**.
**Never quote** the 12 train-split demo tiles (mean 0.739 m) as accuracy — the model was
fitted on those, and the train-vs-test gap is a real **19% memorisation gap**.
