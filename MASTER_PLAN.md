# DepthWizard — master plan

Written 4 Sept 2026, 02:00. Supersedes the dataset sections of `FINETUNE_INDIA_PLAN.md`.
Read `HANDOFF.md` and `HANDOFF_SESSION2.md` first; both remain accurate.

Every claim below is either measured in this repo or cited. Where something is not yet
verified, it says **UNVERIFIED** and names the command that will settle it.

---

## 1 · What problem are we actually solving

SIH26175: **one nadir satellite image → a navigable 3D terrain in the browser.**
Scored 50% on DSM accuracy vs LiDAR *across landscapes*, 50% on rendering and UX.

Ritvik owns exactly one box: **image → per-pixel height above local ground (AGL/nDSM),
in metres.** Divyanshu adds terrain to make a DSM. That split is unchanged and correct.

"Across landscapes" is the phrase that broke tonight. Our model was handed a Himalayan
scene and returned **p50 0.3 m, p99 0.7 m, max 1.1 m** — near-zero everywhere. That is
not a failure. Over bare rock and snow there is nothing above local ground, so AGL ≈ 0
is the *correct* answer, and it is a useless terrain answer. The rendered "mountains"
were `heightmap_preview.png` percentile-stretching 1.1 m of noise across the full grey
ramp, which is exactly why SPEC.md says the preview is never a data source.

---

## 2 · The five gaps GAMUS leaves, and which of these datasets close them

| # | gap | closed by | verdict |
|---|---|---|---|
| 1 | **No DTM.** `DSM = DTM + AGL` has never been validated anywhere in this project | US3D: Track 1 AGL + Track 3 DSM on shared tiles | **CLOSED** |
| 2 | **No terrain / mountains.** Urban-only training, and AGL is the wrong quantity for bare relief | Copernicus GLO-30 DEM — *not a model* | **CLOSED, by not using our model** |
| 3 | **No Indian imagery** | DFC2023 Track 2 (New Delhi) | **PARTIAL** — 2 m labels, tile count unknown |
| 4 | **No Indian ground truth to evaluate against** | nothing here | **OPEN** — only a drone survey fixes it |
| 5 | **Predicted heightmaps are soft** (sharpness 0.084–0.133 vs 1.0) | nothing here | **OPEN — and not a data problem.** See §6 |

Two of five close cleanly, one partially, two stay open. Gap 5 is the one worth being
loudest about: **no dataset in this plan makes the output sharper.** Anyone who says
otherwise has not looked at the GSDs.

---

## 3 · Dataset roster — what each actually contains

The question that matters per dataset is **does it carry a height map AND a semantic
map**, because our model has both a height head and an auxiliary 7-class semantic head.

| dataset | imagery | height | quantity | semantic | GSD (img / label) |
|---|---|---|---|---|---|
| **GAMUS** *(have)* | RGB | ✅ AGL | nDSM | ✅ **7-class** | 0.33 / 0.33 |
| **US3D Track 1** | RGB + MSI | ✅ AGL | nDSM | ✅ **CLS**, LAS codes | ~0.30 / 0.5 |
| **US3D Track 3** | RGB + MSI | ✅ **DSM** | **absolute WGS84 Z** | ✅ CLS | ~0.30 / 0.5 |
| **DFC2023 Track 2** | RGB + SAR | ✅ nDSM | nDSM | ⚠️ **buildings only** | 0.5–1.0 / **2.0** |
| **Copernicus GLO-30** | — | ✅ DSM | **absolute**, EGM2008 | ❌ none | — / 30 |
| **Open Buildings 2.5D** | — | ✅ height | AGL, capped 100 m | ⚠️ building presence | — / ~4 effective |
| **Sentinel-2** | RGB 10 m | ❌ | — | ❌ | 10 / — |

### Semantic detail, since this is what was asked

- **GAMUS** — the reference. 0 unlabelled, 1 ground, 2 low-veg, 3 building, 4 water,
  5 road, 6 tree. `heightmap/data/classes.py` is the authority.
- **US3D** — CLS present for every track, but as **LAS classification codes**, not GAMUS
  ids. LAS 6 is *building*; GAMUS 6 is *tree*. Feeding them through unmapped would have
  the semantic head scoring buildings against trees for a whole run and raising nothing.
  `heightmap/data/us3d.py:CLS_LAS_TO_GAMUS` maps {2,3,4,5,6,9,17} → GAMUS ids;
  `scripts/inspect_us3d.py` verifies it against the real files rather than trusting me.
  Five of seven GAMUS classes are recoverable — there is no distinct low-veg or road
  (bridge deck → road is a judgement call).
- **DFC2023 Track 2** — the task is *building extraction + height estimation*, so the
  annotation is **buildings only**, as polygons. There is no multi-class land cover.
  (Track 1 has roof *types*, a different taxonomy, and is not our task.)
  **UNVERIFIED:** polygon JSON vs rasterised mask. Settle with
  `unzip -l track2.zip | awk '{print $4}' | grep -oE '\.[a-z]+$' | sort | uniq -c`.
  **This needs no code change to handle:** map building → 3 and everything else → 0,
  which is `model.ignore_index`, so `semantic_loss` (terms.py:120, honours
  `ignore_index`) simply trains on building pixels and skips the rest.
- **Copernicus GLO-30 / Sentinel-2 / Open Buildings** — no semantic labels at all. They
  are not training data for our model; see §4.

### Two traps in this table

**Copernicus GLO-30 is a DSM, not a DTM** — ESA's own wording: it represents the surface
*including buildings, infrastructure and vegetation*. Over bare mountains that is
harmless (DSM ≈ DTM). **Over cities it sits at rooftop level, so adding our AGL to it
double-counts every building.** Identical to the SRTM trap already recorded in
`HANDOFF_SESSION2.md` §3. Divyanshu must be told this explicitly.

**DFC2023's label is 4× coarser than its image** — 2 m nDSM on 0.5 m imagery, from
Gaofen-7 / WorldView stereo, not LiDAR. And the contest's *own outcome paper* documents
spatial misalignment between footprints and the stereo nDSM. So it teaches absolute
height and bias; it must be **masked out of the multi-scale gradient loss**, or it will
push sharpness further in the wrong direction.

---

## 4 · The architecture: two paths, chosen by scene

There is **one model**. There are two data paths, and the second one has no model in it.

```
                 ┌─ urban / built-up, 0.165–0.65 m/px
 georeferenced   │     RGB → our AGL model → AGL (metres)
 GeoTIFF  ───────┤     DTM from a bare-earth source (NOT COP30 over cities)
 (CRS+transform) │     DSM = DTM + AGL
                 │
                 └─ mountain / natural, 10–30 m/px
                       COP30 DEM warped to the image grid → DSM directly
                       AGL ≈ 0, so DSM = DTM = DEM.  No model involved.
```

`DSM = DTM + AGL` holds in both; the mountain case is the degenerate one. Tonight's
`max 1.1 m` is the model correctly reporting that degeneracy.

**Why georeferenced input matters, concretely.** The Himalaya run was meaningless
because I had to *declare* `--source-gsd 0.33` as a known fiction against a real ~20–40
m/px. A GeoTIFF carries CRS + affine transform, so GSD is read, the footprint is known
(therefore the DEM tile is fetchable), and the output can carry the same georeferencing.
`heightmap/export.py:41` already accepts `crs`/`transform` and writes them when present
— it passes `None` today only because GAMUS ships ungeoreferenced. Wiring, not a rewrite.

**Georeferencing does not make our AGL model work on mountains.** 10–30 m/px is ~50×
outside the trained 0.165–0.65 band, and AGL over bare rock is still ~0. The GeoTIFF
unlocks the DEM path, not the model path.

### Does this solve the GAMUS problem?

For **terrain**: yes, and honestly — the problem statement itself names SRTM as a
supplementary DEM, so using a DEM for terrain is what the brief asks for, not a
workaround. Training a terrain predictor that would be *worse* than a free 30 m DEM, in
order to avoid using a DEM we are told to use, is negative value.

For **"across landscapes"** in the scoring: the two-path system covers urban and natural.
What it does not cover is anything where AGL matters *and* we have no Indian ground
truth — gap 4, still open.

---

## 5 · Ordered plan

| # | task | why now | cost | gate |
|---|---|---|---|---|
| **0** | **Mode-vs-mean decode test** on existing A7 weights | §6. Free, no retraining, attacks the one open problem no dataset touches | 30 min write, minutes to run | none |
| **1** | `scripts/fetch_dem.py` — GeoTIFF → COP30 → warped bundle | Makes the Himalaya scene actually work; unblocks Aakarsh's terrain render today | ~40 min, no GPU | none |
| **2** | US3D: derive `DTM = DSM − AGL`; score COP30/SRTM against it over JAX+OMA | First real number on the team's central untested assumption | 1 day, no GPU | US3D download |
| **3** | Zero-shot v0.1.0 on US3D Track 1 | Second held-out-domain number, on a different **sensor**, free | 2 h GPU | US3D download |
| **4** | DFC2023: inspect, check misalignment, write loader | Go/no-go on the only Indian labelled source | 1 day | track2.zip |
| **5** | Fine-tune **ours** (not DAv2) on GAMUS+US3D+Delhi, mixed batches | The India model | ~6 h GPU | 3, 4 |
| **6** | Drone survey + ODM, downsampled to trained GSD | The only honest Indian evaluation set | days | — |

Machinery for task 5 is **already built and tested tonight**:
`heightmap/data/mixing.py` (exact per-batch domain ratios, with guards that raise rather
than silently drop a domain), `heightmap/data/us3d.py`, and block-level surgery in
`net.py` + `configs/finetune/`.

### Ours or DAv2 for the India fine-tune? — Ours, decisively

- DFC2023's nDSM **is** our output quantity, in metres. Drop-in.
- DAv2's loss is affine-invariant by construction. A0 measured the cost: global affine
  r 0.405, per-tile *oracle* affine r 0.817. That gap is unreachable by rescaling, which
  is precisely why the depth head was replaced rather than calibrated.
- Ours already has bins, FiLM on log(GSD), semantic aux and metric losses tuned to this
  target. Fine-tuning DAv2 to metric nDSM means rebuilding what exists.

---

## 6 · The softness problem — and why no dataset here fixes it

Sharpness ratio (`metrics.py:73` — gradient at **true-edge** locations; mean gradient
cannot detect blur, that was bug #5) sits at **0.084–0.133**. Truth is 1.0.

Calibration by degrading the ground truth: 1 px shift → 0.500, σ=1 → 0.398, σ=2 → 0.202,
**σ=3 → 0.135**. So the output is equivalent to truth blurred at **σ ≈ 3 px ≈ 2 m**, and
the metric is not saturated — 0.13 is genuinely poor with real headroom.
*(That calibration is from synthetic slabs; re-running it on real GAMUS is outstanding.)*

Four approaches have failed to move it: guided filter (0.123), semantic flattening
(0.108, and MAE got worse), a learned full-res refiner (0.133), the deployed tiled path
(0.125–0.133).

**No dataset in §3 addresses this, and it is important to say why rather than hope:**

- **US3D's AGL is 0.5 m** — *coarser* than GAMUS's 0.33 m. Sharper it is not.
- **DFC2023's nDSM is 2 m** — 4× coarser than its own imagery. Training on it without
  masking it out of the gradient loss would make sharpness measurably **worse**.
- **COP30 is 30 m.** Not in the conversation.

**Sharpness is an architecture and decode problem, not a data problem.**

### The one untested fix, and it is free

`heads.py:110`:
```python
h = (p * centres[:, :, None, None]).sum(dim=1, keepdim=True)
```
That is the **conditional mean** of the bin distribution. The entire argument for the
bins head (HANDOFF §3) is that at a roof edge the truth is bimodal — 0 m or 18 m — and
a regression head is forced to emit ~9 m, a value in mid-air, which *is* the melted-edge
failure. But **the expectation of a bimodal distribution is also 9 m.** The head
represents the bimodality correctly and then discards it on the last line.

This explains why A2's bins changed nothing, why sharpness is identical across A1/A2/A6,
and why no post-process recovers it: the information is destroyed inside the head.

**A7 already emits `bin_logits`.** Decode with the mode (`centres[argmax(p)]`) or a
temperature-sharpened softmax instead of the mean — same weights, different last line.
If sharpness jumps from 0.13 toward 0.3–0.4 with building MAE intact, that is the answer
and it costs nothing. **This is task 0 and should be done before any dataset work.**

One implementation note: `bin_logits` come out at decoder resolution (296²) while
`height` is collapsed *before* the 518² interpolate, so a mode-decode script must decode
at 296² and upsample itself — and nearest vs bilinear there is not neutral for a
sharpness measurement. Report both.

Ranked after that: entropy penalty on `p` during training; an edge-conditioned gradient
loss (the current `multiscale_gradient_loss` avg-pools 3 of its 4 scales, destroying the
fine detail it exists to score); learned upsampling to replace the 296²→518² bilinear;
the `--w-grad 6/20/60` sweep that was queued and never run.

**A diagnosis that was wrong, recorded so it is not repeated:** the blur was argued to be
caused by the 296²→518² bilinear upsample, and a full-resolution refiner was predicted to
break the ceiling. It did not (0.129 → 0.133). σ=3 px at 518² is σ≈1.7 px at 296², so
most of the blur exists *before* the upsample.

**TDA cannot sharpen edges.** Persistence measures the *prominence* of features, and a
sharp step and a σ=3-blurred step have essentially identical persistence diagrams — it is
invariant to exactly the quantity being measured. TDA *can* do persistence-thresholded
watershed for building instance separation (which would fix the semantic-flattening
failure, where `ndi.label` merged touching footprints and *erased* true edges), cleanup
of ringing after aggressive sharpening, and bottleneck distance as a diagnostic for how
many small structures the blur erased. Instance separation is a clean ask for Vineeth.

---

## 7 · What we are deliberately not doing

- **Not training a terrain/DEM predictor.** A free 30 m DEM beats anything we could train
  in the time available, and the problem statement names SRTM as a supplementary DEM.
- **Not adding a DSM head.** Absolute elevation is not recoverable from one nadir image
  without external terrain, and it is Divyanshu's box. AGL being offset-free is *why* it
  transfers: held-out NYC kept building MAE at 1.585 m while overall bias blew out to
  −2.02 m.
- **Not using GlobalBuildingAtlas heights.** Validated in India against ICESat-2 across
  8 cities: **MAE 30.3 m, RMSE 41.8 m.** That is 20× our current error. Footprints are
  excellent (>99% completeness) and worth keeping for masks; the heights are unusable.
- **Not using OSM `building:levels`.** 4.6% of buildings globally carry it, **below 1%**
  in South Asia, with Pune and Ranga Reddy among the worst anywhere.
- **Not reviving the shadow loss** without re-running its control. It is a measured
  negative (0.4% at w=0.1 over 38% of tiles) and India's low-latitude, high-sun regime
  makes shadows *shorter* — i.e. the cue is weaker there, not stronger.
- **Not quoting the 12 train-split demo tiles** (mean building MAE 0.739 m) as accuracy.

---

## 8 · Honest claims for the report

- Held-out city (NYC, never trained on): **building MAE 1.585 m, r 0.901.** Lead with it.
- In-domain: 1.549 m. Content-rich median on unseen test tiles 1.600 m; val agrees at
  1.560 m. Worst tile in the unseen pool: **46.6 m**.
- Train-pool median 1.339 vs test 1.600 → a **19% memorisation gap**.
- **A1–A5 sit in a 1.769–1.848 band on a single seed. Those configurations cannot be
  ranked from that data.** A7 (backbone scale) is the only unambiguous win.
- The shadow loss is a **clean negative result on a properly controlled comparison**.
- **Nothing has been validated on Indian data.** External runs are qualitative only.
- Honest target: **2–4 m MAE urban**; for India specifically, published deep-learning
  models in Global South cities land at **2.2–7.0 m MAE**, so **3–5 m with r > 0.8** is
  the defensible claim until a drone survey exists.
