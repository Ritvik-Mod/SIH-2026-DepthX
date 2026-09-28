# SIH26175 DepthWizard — problem statement vs what we have

Written 4 Sept 2026. Companion to `PIPELINE.md` (what is built). This file is the honest scorecard: **what the PS asks, what
exists, what does not, and what we have deliberately decided not to do.**

Written to be handed to a fresh agent or a reviewer without further context.

---

## 1 · What the PS asks, in plain language

Feed in **one ordinary optical satellite image**. Get out **the height of everything in
it** — ground and structures — as a proper geospatial file. Then let a user fly through
it in 3D in a browser.

Two input modes:
- **PNG / JPG** (no spatial metadata) → a **relative** DSM (rDSM)
- **GeoTIFF** (has coordinate metadata) → an **absolute** DSM in real metres, calibrated
  using a coarse DEM such as SRTM, or a few Ground Control Points

Suggested method: use a pre-trained **monocular depth backbone** to get relative depth,
then a **scale-calibration** module to convert it to absolute elevation.

**Scoring: 50% DSM accuracy** (RMSE, MAE, correlation vs LiDAR) *with stability across
**urban, sparse, hilly and forested** landscapes* — **50% rendering and UX.**

---

## 2 · Requirement-by-requirement scorecard

| # | PS requirement | Status | Detail | Owner |
|---|---|---|---|---|
| 1 | Pre-trained monocular depth backbone | ✅ **done** | DINOv2 ViT-L/14 + DPT neck from Depth Anything V2 — ~99% of parameters retained | Ritvik |
| 2 | Backbone → initial **relative** depth, then calibrate | ⚠️ **deviated, measured** | We predict metric directly. Justified by A0; see §3 | Ritvik |
| 3 | **Scale-calibration** module (DEM / GCP / scene stats) | 🔄 **partial** | FiLM on log(GSD) gives metres directly. DEM/GCP calibration module not built | Ritvik + Divyanshu |
| 4 | Output an **absolute DSM** | ❌ **not yet** | We emit **AGL**. `DSM = DTM + AGL`; the terrain-add stage is not wired in | Divyanshu |
| 5 | Non-geo PNG/JPG → **rDSM** | ❌ **not yet** | We emit metric AGL. For unknown-GSD input those metres are conditional on a declared guess | Ritvik |
| 6 | Geo GeoTIFF → absolute DSM, **CRS preserved** | ❌ **not yet** | `export.py:41` accepts `crs`/`transform`, currently passes `None` | Ritvik |
| 7 | Accuracy across **urban / sparse / hilly / forested** | ❌ **1 of 4** | See §4 — **this is the largest scoring gap** | Ritvik |
| 8 | Output in a **standard geospatial format** | ✅ **done** | float32 GeoTIFF + JSON sidecar, DEFLATE, tiled, nodata `-9999` | Ritvik |
| 9 | Project texture onto 3D mesh, **Three.js** | 🔄 **in progress** | Codebase on `main`; bundles delivered | Aakarsh |
| 10 | **First-person navigation** | 🔄 in progress | | Aakarsh |
| 11 | UI: **upload imagery**, visualise, **validate vs reference** | ❌ not started | Upload path and the pred-vs-truth comparison view | Aakarsh |
| 12 | **Standalone deployable** app + docs | 🔄 partial | `PIPELINE.md` is the technical doc; packaging not done | team |
| 13 | RMSE / MAE / **correlation** reported | ✅ done | All three, plus per-band bias and building-only breakdowns | Ritvik |

**Summary: 4 done, 4 in progress, 5 not started.** The model half is the strongest part;
the gaps are coverage, packaging and the DSM assembly.

---

## 3 · The one real methodological deviation, and its defence

The PS suggests: backbone → relative depth → calibrate to absolute.
We kept the backbone and **replaced the depth head** with a metric one.

**Why, in one line: calibration fixes scale, not shape — and the shape is what is wrong.**

Depth Anything V2 is trained with an **affine-invariant** loss: `a·d + b` scores
identically to `d`. Absolute scale was not lost, it was *optimised away by design*.

Now the decisive part. **Pearson correlation is invariant under `a·d + b`.** So if the
correlation is poor after fitting the best possible affine, no calibration whatsoever
can improve it. We measured exactly that (`scripts/zero_shot_eval.py`, A0, 400 tiles):

| fit | building MAE | **r** |
|---|---|---|
| single global affine — *what a real SRTM/GCP deployment achieves* | 4.830 m | **0.405** |
| per-tile **oracle** affine — *cheating, fitted against the truth itself* | 3.153 m | 0.817 |
| **our metric model, on a city it never trained on** | **1.585 m** | **0.901** |

Because `r` is affine-invariant, **0.405 is also stock DAv2's raw uncalibrated
correlation** — fitting the affine improved MAE and moved `r` by exactly zero.

Per-band bias on the global fit: **−9.9 m** at 10–30 m, **−26.6 m** above 30 m. It
flattens tall structures systematically, and no single `(a, b)` repairs a field that is
wrong by different amounts at different heights.

**We did not ignore the brief's method. We implemented it, measured its ceiling, and
exceeded it.** That distinction belongs in the report and in the UI.

### How we intend to close even this gap
1. Ship the PS pipeline as a **selectable branch** (`--backend relative`): stock DAv2 →
   relative depth → calibrate → DSM, emitting the same bundle.
2. Calibrate that branch from **SRTM / GCPs**, not from LiDAR truth. A0 currently fits
   against the truth, which is *generous* — the honest PS-method number will be **worse**
   than 4.830 m, strengthening the argument.
3. Emit **rDSM** for non-georeferenced input (requirement #5), which the relative branch
   gives for free.

---

## 4 · The biggest gap: landscape coverage is 50% of the marks

The PS scores "performance stability across urban, sparse, hilly, and forested."

| landscape | status | evidence |
|---|---|---|
| **urban** | ✅ strong | held-out city building MAE **1.585 m**, r **0.901** |
| **sparse** | ❓ never measured | GAMUS has low-density tiles but we have never reported the category |
| **hilly** | ❌ **fails** | Himalayan scene returned p50 0.3 / max **1.1 m** — nothing |
| **forested** | ❓ never measured | GAMUS has a tree class but no forest-scene category |

**Why hilly fails, and it is not a bug.** Our model outputs AGL — height above *local
ground*. Over bare rock and snow there is nothing above local ground, so **AGL ≈ 0 is the
correct answer** and a useless terrain answer. What a mountain scene needs is the terrain
elevation itself, which is a different quantity, owned downstream.

**Two fixes, both cheap:**

1. **Stratified evaluation, free, today.** GAMUS tiles carry per-pixel classes. Stratify
   the existing unseen test tiles by building-fraction and tree-fraction and report MAE
   per stratum. That converts *sparse* and *forested* from ❓ into numbers using data
   already on the cluster, with no downloads and no training.
2. **DEM path for hilly.** For bare terrain `DSM ≈ DTM ≈ DEM`, and Copernicus GLO-30 is
   free, global and georeferenced. **The PS explicitly names SRTM as a supplementary
   DEM**, so this is the prescribed calibration route, not a workaround.

---

## 5 · The architecture we are converging on

**One model. Two data paths, chosen by scene type. The second has no model in it.**

```
                 ┌─ urban / built-up, 0.165–0.65 m/px
 georeferenced   │     RGB → HeightNet → AGL (metres)
 GeoTIFF   ──────┤     + terrain (flat plane at tile scale, or DEM trend at scene scale)
 (CRS+transform) │     → DSM
                 │
                 └─ mountain / natural, 10–30 m/px
                       COP30 DEM warped to the image grid → DSM directly
                       AGL ≈ 0.  No model involved.
```

### Two traps recorded so they are not rediscovered

**Copernicus GLO-30 and SRTM are DSMs, not DTMs.** ESA's own wording: the surface
*including buildings, infrastructure and vegetation*. Over bare mountains that is
harmless (DSM ≈ DTM). **Over cities they sit at rooftop level, so adding our AGL
double-counts every building** — an 18 m building becomes 36 m and the render still looks
plausible. There is no free global bare-earth DTM at useful resolution.

**Resolution mismatch.** COP30 is 30 m; our AGL is 0.33 m. One DEM pixel covers ~90×90
image pixels, so a DEM base under an urban tile is a coarse staircase whose edges read as
terrain features that do not exist. For a single 338 m tile, **a flat base plane is more
correct than an SRTM base.**

---

## 6 · Dataset roster — what each actually contains

The question that matters is whether a dataset carries **both** a height map and a
semantic map, because the model has both a height head and a 7-class semantic head.

| dataset | imagery | height | quantity | semantic | GSD img / label |
|---|---|---|---|---|---|
| **GAMUS** *(in use)* | RGB | ✅ AGL | nDSM | ✅ **7-class** | 0.33 / 0.33 |
| **US3D Track 1** | RGB + MSI | ✅ AGL | nDSM | ✅ CLS (LAS codes) | ~0.30 / 0.5 |
| **US3D Track 3** | RGB + MSI | ✅ **DSM** | **absolute WGS84 Z** | ✅ CLS | ~0.30 / 0.5 |
| **DFC2023 Track 2** | RGB + SAR | ✅ nDSM | nDSM | ⚠️ **buildings only** | 0.5–1.0 / **2.0** |
| **Copernicus GLO-30** | — | ✅ DSM | absolute (EGM2008) | ❌ | — / 30 |
| **Open Buildings 2.5D** | — | ✅ height | AGL, capped 100 m | ⚠️ presence only | — / ~4 effective |

**US3D is the most valuable addition** — the only public source pairing GAMUS's exact
triple with a genuine **sensor** change (WorldView-3 vs aerial orthophoto) *and* clean
LiDAR truth, plus an absolute DSM so that `DTM = DSM − AGL` is derivable. It is the only
place `DSM = DTM + AGL` can be validated at all.

Its CLS uses **LAS classification codes, not GAMUS ids** — LAS 6 is *building*, GAMUS 6
is *tree*. Unmapped, the semantic head would score buildings against trees for a whole
run and raise nothing. `heightmap/data/us3d.py` maps them and
`scripts/inspect_us3d.py` verifies the mapping against the real files.

**DFC2023 carries nDSM, not DSM** — the same quantity we already predict — and its label
is **2 m against 0.5 m imagery**, from stereo rather than LiDAR, with spatial
misalignment documented in the contest's own outcome paper. It teaches absolute height
and bias; it must be **masked out of the multi-scale gradient loss** or it will make
sharpness worse. It contains **New Delhi**, which is its entire value to us.

---

## 7 · The softness problem — open, and not a data problem

Sharpness ratio (`metrics.py:73`, gradient measured at **true-edge** locations; mean
gradient cannot detect blur — that was bug #5) sits at **0.084–0.133**. Truth is 1.0.

Calibrating by degrading the truth: 1 px shift → 0.500, σ=1 → 0.398, σ=2 → 0.202,
**σ=3 → 0.135**. The output is equivalent to **truth blurred at σ ≈ 3 px ≈ 2 m**, and the
metric is not saturated — there is real headroom.

Four approaches failed: guided filter (0.123), semantic flattening (0.108, MAE worse),
learned full-resolution refiner (0.133), the deployed tiled path (0.125–0.133).

**No dataset above fixes this.** US3D's AGL is 0.5 m — *coarser* than GAMUS's 0.33 m.
DFC2023's is 2 m. COP30 is 30 m. It is an architecture and decode problem.

**The leading hypothesis, untested and free.** `heads.py:110`:
```python
h = (p * centres[:, :, None, None]).sum(dim=1, keepdim=True)
```
That is the **conditional mean** of the bin distribution. The entire argument for the
bins head is that at a roof edge the truth is bimodal — 0 m or 18 m — and a regression
head is forced to emit ~9 m, a value in mid-air, which *is* the melted-edge failure.
**But the expectation of a bimodal distribution is also 9 m.** The head represents the
bimodality correctly and discards it on the last line. This explains why bins changed
nothing, why sharpness is identical across A1/A2/A6, and why no post-process recovers it.

A7 already emits `bin_logits`. Decode with the **mode** or a temperature-sharpened
softmax — same weights, one line. Ranked after that: entropy penalty on `p`;
edge-conditioned gradient loss (the current one avg-pools 3 of its 4 scales, destroying
the detail it exists to score); learned upsampling; the `--w-grad` sweep never run.

**Recorded so it is not repeated:** the blur was argued to come from the 296²→518²
bilinear upsample, and a full-resolution refiner was predicted to break it. It did not
(0.129 → 0.133). σ=3 px at 518² is σ≈1.7 px at 296², so most of the blur exists *before*
the upsample. **TDA cannot sharpen edges** either — a sharp step and a σ=3-blurred step
have essentially identical persistence diagrams. TDA *can* do persistence-thresholded
watershed for building instance separation, which is a clean ask for Vineeth.

---

## 8 · What we are deliberately NOT doing, and why

| decision | reason |
|---|---|
| **Not training a terrain/DEM predictor** | A free 30 m DEM beats anything trainable in the time available, and the PS names SRTM as a supplementary DEM |
| **Not adding a DSM head** | Absolute elevation is not recoverable from one nadir image without external terrain. AGL being offset-free is *why* it transfers — held-out NYC kept building MAE at 1.585 m while overall bias blew out to −2.02 m |
| **Not fine-tuning DAv2 instead of ours** | Its affine-invariant objective is exactly what A0 measured the cost of. Fine-tuning it to metric nDSM rebuilds what already exists |
| **Not using GlobalBuildingAtlas heights** | Validated in India against ICESat-2 across 8 cities: **MAE 30.3 m, RMSE 41.8 m** — 20× our current error. Footprints (>99% complete) are still useful for masks |
| **Not using OSM `building:levels`** | 4.6% of buildings globally, **below 1%** in South Asia, with Pune and Ranga Reddy among the worst anywhere |
| **Not using CartoDEM as a label** | Open tier is 30 m; the 2.5 m product is priced for non-government and states **8 m LE90** vertical accuracy, five times our error |
| **Not reviving the shadow loss** without its control | Measured negative (0.4% at w=0.1 over 38% of tiles). India's high sun makes shadows *shorter*, i.e. the cue weaker |
| **Not quoting train-split demo tiles** | The 12 curated train tiles average 0.739 m; the model was fitted on them. Train-vs-test gap is a real **19% memorisation gap** |

---

## 9 · Priority order to close the gaps

| # | task | closes | cost | blocked on |
|---|---|---|---|---|
| 1 | **Stratified landscape evaluation** on existing GAMUS test tiles | #7 — turns *sparse* and *forested* into numbers | hours, no download | nothing |
| 2 | **Mode-vs-mean decode test** | §7 softness — improves the 50% accuracy half | 30 min, no retraining | nothing |
| 3 | **GeoTIFF in / GeoTIFF out**, CRS preserved | #6 | ~half day | nothing |
| 4 | **rDSM output mode** for non-geo input | #5 | ~30 min | nothing |
| 5 | **`fetch_dem.py`** — GeoTIFF → COP30 → warped bundle | #4, #7 hilly | ~1 h | nothing |
| 6 | **Relative-depth baseline branch** with SRTM/GCP calibration | #2, #3 | ~3 h | nothing |
| 7 | US3D: derive `DTM = DSM − AGL`, score DEMs against it | #4 validation | 1 day | download |
| 8 | DFC2023 inspect + misalignment check + loader | India | 1 day | download |
| 9 | Fine-tune on GAMUS + US3D + Delhi, mixed batches | India | ~6 h GPU | 7, 8 |
| 10 | Drone survey + ODM | Indian ground truth — the only route | days | — |

**Items 1–6 need no downloads and no new data.** They are also the ones that move the
scored criteria most directly. Items 7–10 depend on data still arriving.

---

## 10 · Honest claims for the report

- Held-out city (NYC, never trained on): **building MAE 1.585 m, r 0.901.** Lead with it.
- In-domain: 1.549 m. Unseen-test content-rich median 1.600 m; val agrees at 1.560 m.
- Worst tile in the unseen pool: **46.6 m**.
- Train-pool median 1.339 vs test 1.600 → **19% memorisation gap**.
- **A1–A5 sit in a 1.769–1.848 band on a single seed and cannot be ranked from that
  data.** A7 (backbone scale) is the only unambiguous win.
- The shadow loss is a **clean negative result on a properly controlled comparison** —
  report it, do not bury it.
- **Nothing has been validated on Indian data.** External runs are qualitative only.
- Honest target: **2–4 m MAE urban**. For India, published deep-learning models in
  Global South cities land at **2.2–7.0 m MAE**, so **3–5 m with r > 0.8** is the
  defensible claim until a drone survey exists.
