# India fine-tuning — architecture plan and dataset survey

Written 4 Sept 2026. Read `HANDOFF.md` and `HANDOFF_SESSION2.md` first.

Every number below is either read out of this repo's own files, or cited to a paper /
dataset page. Where I could not verify something, it says so.

---

## 0 · Correcting the premise

**We never fine-tuned "2 blocks."** There is no per-block freezing anywhere in this
codebase. What exists is three *parameter groups* at three learning rates
(`heightmap/models/net.py:param_groups`, `configs/base.yaml:train`):

| group | module | lr | trained? |
|---|---|---|---|
| encoder | the **entire** DINOv2 backbone | `1.0e-5` | yes, all blocks |
| decoder | the **entire** DPT neck | `5.0e-5` | yes |
| head | FiLM + bins + semantic (+ unc) | `1.0e-4` | yes, from scratch |

`freeze_encoder_epochs: 2` freezes the whole backbone for two epochs, then unfreezes
**everything**. So A1–A7 trained all 24 ViT-L blocks — just 10× slower than the head.

The "2 blocks" idea probably comes from the docstring at the top of `net.py`, which is a
statement about *what we inherited*, not about freezing:

```
backbone  keep                      general visual features
neck      keep structure, retrain   fusion is right, weights are depth-tuned
head      discard                   encodes perspective geometry + affine invariance
```

Two of those three are reused, one is thrown away. That is the "2 vs 3" you remember.

### There is no "first block that finds edges" in a ViT

That is a CNN intuition and it does not carry over. In a CNN, stage 1 is high-resolution
and small-receptive-field, so it genuinely is the edge stage. In DINOv2, **every block
operates on the same token grid at the same resolution** — 518/14 = 37×37 tokens, block
1 through block 24. There is no spatial hierarchy in the encoder at all.

The hierarchy is **manufactured by the DPT neck**, and this is the part that matters.
Verified from the actual config files:

| | DAv2-Small (A1–A6) | DAv2-Large (A7, our best) |
|---|---|---|
| blocks | 12 | 24 |
| hidden | 384 | 1024 |
| `out_indices` (taps) | `[3, 6, 9, 12]` | `[5, 12, 18, 24]` |
| `reassemble_factors` | `[4, 2, 1, 0.5]` | `[4, 2, 1, 0.5]` |
| `neck_hidden_sizes` | `[48, 96, 192, 384]` | `[256, 512, 1024, 1024]` |
| `fusion_hidden_size` | 64 | 256 |

Small config read from the local HF cache; Large config fetched from the model repo.

Read the two rows together. **Block 5's tokens get `reassemble_factor 4` — upsampled 4×
into the finest-resolution branch of the neck. Block 24's get `0.5` — downsampled into
the coarsest, most semantic branch.**

So the early blocks *are* the fine-detail path, not because they compute edges but
because DPT assigns them to the high-resolution branch. Freezing block 5 does not
"leave edge detection alone" — it freezes the input to the sharpest thing the decoder
has. Given that sharpness is our headline open problem (§4 of the session-2 handoff),
that is exactly the wrong thing to freeze by accident.

---

## 1 · Which blocks to tune for India

### The evidence

**Lee et al., "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts",
ICLR 2023** (arXiv 2210.11466) is the paper that answers this question directly, and its
finding is counter-intuitive enough to be worth stating plainly:

| shift type | best block to tune | examples in the paper |
|---|---|---|
| **input-level** (corruption, sensor, appearance) | **first** | CIFAR-10-C, ImageNet-C |
| feature-level | middle | Living-17, Entity-30 |
| output-level (label distribution) | last | Waterbirds, CelebA |

On CIFAR-10-C, tuning **only the first block and freezing everything else beats full
fine-tuning by ~3%**. Their explanation: fine-tuning more parameters on a small target
set destroys pretrained information, and which information you can afford to lose
depends on where the shift is.

They also give an automatic selection criterion so you don't have to guess —
**relative gradient norm**, `RGN = ‖g‖₂ / ‖θ‖₂` per layer, tune the layers with the
highest RGN. Auto-RGN was their best single-run method and adds no hyperparameters.

### What kind of shift is India?

**It is mixed, and that is the honest answer.**

*Input-level* — different sensor and processing chain, heavier aerosol/haze, different
roof materials (concrete, blue/red-painted metal, tarpaulin, tile vs North American
asphalt shingle), different vegetation spectra, and a lower-latitude sun regime.
GAMUS is a leaf-off orthophoto survey at 39–41°N with elevation p50 40°; Indian
acquisitions run much higher, so shadows are shorter and the dominant physical height
cue weakens. → argues for **early** blocks.

*Output-level* — Indian building-height statistics differ from DC/PHL/NYC: more
2–4 storey stock, different tall-building density. That is a shift in `p(height)`.
→ argues for **late** blocks and the bins head.

*Feature-level* — denser, finer-grained, more irregular footprints; narrow gullies;
much less of the regular rectilinear grid the sun-angle work relied on.
→ argues for **middle**.

Anyone who tells you confidently which one dominates is guessing. **We should measure
it, and the measurement is cheap.**

### Recommendation

**Do not pick blocks by argument. Run Auto-RGN, then run a 4-arm block sweep, then
decide.** But if you need a prior to start from, it is: *early blocks matter more than
you think, and the current "train everything at 1e-5" is probably wrong for a small
Indian set* — because with a few hundred Indian tiles against 5,004 GAMUS tiles,
full fine-tuning at any LR is the configuration most likely to forget GAMUS.

---

## 2 · The fine-tuning design

### 2.1 Measure the shift before choosing (half a day)

Add a `scripts/rgn_probe.py` that loads `outputs/a7_vitl/best.pt`, runs one epoch's
worth of forward+backward on the Indian set **without stepping the optimiser**, and
reports per-block `‖g‖₂ / ‖θ‖₂` for all 24 ViT-L blocks plus the four neck fusion
stages. That plot alone tells you where the model thinks the shift is. It costs one
epoch of compute and no training.

Sanity anchor: run the same probe on held-out GAMUS. The *difference* between the two
RGN curves is the signal; the absolute curve is dominated by depth-dependent gradient
scaling and is not interpretable on its own.

### 2.2 The block sweep — four arms, ViT-S first

Ablation discipline from the A-ladder applies. Run these on **ViT-S** to rank them (1.9×
throughput, and §1 of the session-2 handoff already established ViT-S ranks configs
fine), then scale the winner to ViT-L.

| arm | trainable | rationale |
|---|---|---|
| **F0** control | heads only, everything else frozen | the floor. If this matches the others, the shift is output-level and we are done cheaply. |
| **F1 early** | blocks 1–8 + heads | the surgical-FT prior for input-level shift; also unfreezes the block-5 tap that feeds the finest DPT branch |
| **F2 late** | blocks 17–24 + neck + heads | what you assumed we did; the output-level arm |
| **F3 full** | everything at current LRs | today's recipe, as the baseline to beat |

Add **F4 = Auto-RGN top-k** once §2.1 has run, with k chosen so it trains a comparable
parameter count to F1/F2 — otherwise the comparison is confounded by capacity.

**Every arm needs a seed repeat.** The A1–A5 lesson is in the handoff in bold: a
1.769–1.848 band on a single seed is not a ranking. Two seeds minimum, and report the
spread, or this sweep produces the same unusable result.

### 2.3 Keep GAMUS in every batch — and the reason is measurable

HANDOFF §9 already says this; here is the mechanism. Our held-out-NYC result is
building MAE 1.585 with **overall bias −2.02 m**. That offset appeared on a city in the
*same country* with the *same sensor*. A few hundred Indian tiles fine-tuned alone will
produce a much larger version of that, in the opposite direction, and will destroy the
1.549 GAMUS number we are reporting as our headline.

Concrete: **mixed batches at a fixed ratio**, not sequential fine-tuning. Start at
75% GAMUS / 25% India per batch and treat the ratio as a swept hyperparameter. With
batch 8 that is 6 + 2, which is coarse; use `accum_steps` to get an effective batch of
16–32 so the ratio is actually realised rather than rounded.

Report the GAMUS number after every India run. A fine-tune that improves India by 1 m
and costs 0.5 m on GAMUS is a real trade, but it has to be visible to be traded.

### 2.4 Discriminative LRs, layerwise

Replace the flat `lr_encoder` with a layerwise decay, standard for ViT fine-tuning:
`lr_block_i = lr_encoder · γ^(N−i)` with γ ≈ 0.65–0.75, N = 24. This is a strictly more
expressive version of the current setup (γ = 1 recovers today's behaviour) and is the
natural way to express "early blocks move less" or, if RGN says otherwise, the reverse.
It goes in `param_groups()` and is maybe 20 lines.

### 2.5 What NOT to touch

- **The bins head's `h_max: 250` and 128 bins.** Indian stock is *shorter*, not taller.
  Bin centres are predicted per-image by the MLP, so they adapt on their own.
- **The FiLM GSD conditioning.** This is the one part of the architecture that is
  *already* the India adaptation: it is why `predict_external.py` resamples to the
  trained band. Trained coverage is 0.165–0.65 m/px. Cartosat-3 pan at 0.25 m is inside
  it; Cartosat-2S at 0.6–1.0 m is at or past the edge. State that as a limitation.
- **The shadow loss.** It is a measured negative result on GAMUS (0.4%, §1 of the
  session-2 handoff). Do not resurrect it for India on the theory that "Indian shadows
  are different" without first running the control arm again. It would also need sun
  angles re-estimated for a low-latitude, high-elevation regime where shadows are
  short — i.e. where the cue is *weakest*.

---

## 3 · Dataset survey — Indian data

**Headline: there is no high-resolution Indian building-height benchmark. One usable
labelled Indian source exists, and it is small.**

### 3.1 DFC2023 Track 2 — the only directly Indian labelled option ✅

IEEE GRSS Data Fusion Contest 2023, "Large-Scale Fine-Grained Building Classification
and Height Estimation".

- **New Delhi is one of the 11 Track-2 cities** (with Berlin, Barcelona, Rio, Sydney,
  Brasilia, Copenhagen, New York, San Diego, Sao Luis, Portsmouth).
- 2,957 tiles of **512×512** across all cities; 125,153 annotated buildings.
  Per-city counts are not published — **New Delhi's share is unknown and must be
  counted after download.** Do not plan around a number until you have it.
- Optical: SuperView-1 **0.5 m**, Gaofen-2 0.8 m, Gaofen-3 1 m.
- Labels: **nDSM at ~2 m GSD**, from Gaofen-7 / WorldView-1&2 stereo — plus building
  polygons and roof-type classes.
- Download: `ieee-dataport.org`, registration + contest terms.

**Fit assessment, honestly:**
- ✅ Same quantity as ours — nDSM, i.e. AGL. Drop-in for our target.
- ✅ 0.5 m optical is **inside** our trained GSD band (0.165–0.65).
- ⚠️ 512² tiles vs our 518 crop. One pixel short — you will need reflect-padding or a
  510 crop for this source. Trivial but it will bite silently if unnoticed.
- ❌ **The label is 2 m GSD against 0.5 m imagery — 4× coarser.** This cannot teach
  edge sharpness; a 2 m label has already blurred exactly the structure §4 of the
  session-2 handoff is chasing. Use it for **absolute height and bias**, and mask it out
  of the gradient loss, or it will actively make sharpness worse.
- ❌ Stereo-derived nDSM, not LiDAR. Noisier than GAMUS truth, with its own artefacts.

**Verdict: take it. It is the only real Indian supervision available, but weight it for
what it is — a bias/scale teacher, not a structure teacher.** That maps exactly onto the
SILog-vs-L1 and correlation-vs-MAE split in §1 of HANDOFF.md.

### 3.2 Google Open Buildings 2.5D Temporal — best weak-label source ⚠️

- Covers **India explicitly**, plus Africa, SE Asia, Latin America. 2016–2023 annual.
- Rasters at **0.5 m**, but **effective resolution ~4 m** — it is Sentinel-2 derived
  (10 m native). The 0.5 m grid is interpolation, not information.
- Heights **capped at 100 m**, relative to ground (so: AGL, same quantity as ours).
- Licence CC-BY-4.0 / ODbL-1.0. Earth Engine or direct GCS download.

**The catch, and it is a big one: the quoted 1.5 m MAE was validated in North America,
Europe and Japan only — not in the Global South, which is the entire region the dataset
covers.** Google says this themselves. So the 1.5 m figure is *not* an India number and
must never be quoted as one.

The independent check that does exist: "Narrowing the gap for city building height
predictions" (Sci Rep 2025) evaluated Open Buildings Temporal against 1.5 m DEMs in
Nairobi, Kathmandu and Quito and got **MAE 2.5 m (Nairobi) and 1.2 m (Kathmandu)**.
Kathmandu is the closest published analogue to Indian urban fabric that I could find.

**Verdict: usable as weak supervision at ~4 m effective resolution, i.e. for the same
job as DFC2023 — anchoring absolute height. Never as a sharpness or edge target, and
never as an evaluation set.**

### 3.3 GlobalBuildingAtlas — reject as a label source ❌

HANDOFF.md §9 item 5 proposes "weak labels from GlobalBuildingAtlas". **That should be
dropped.** A 2025 IOP study evaluated GBA specifically in India — Bengaluru, Chennai,
Hyderabad, Kolkata, Ahmedabad, Mumbai, Gurgaon, Pune — validating heights against
ICESat-2:

- **MAE 30.3 m, RMSE 41.8 m**, with systematic underestimation growing with height,
  up to −60% on tall structures.

Our model's held-out error is 1.585 m. Training on a 30 m-MAE label would be training
on noise an order of magnitude larger than our current error. GBA's *footprints* are
excellent (>99% completeness) and are worth keeping for building masks — but the
heights are unusable for us.

### 3.4 OpenStreetMap `building:levels` — reject as primary ❌

Also proposed in HANDOFF §9. The global assessment (Biljecki et al., *Building and
Environment* 2023) measured **4.6% of buildings tagged with `building:levels`, 2.9% with
`height`, 7% with either.** Regional breakdown puts South Asia **below 1%**, and names
Indian administrative units (Ranga Reddy, Telangana; Pune, Maharashtra) among the
*lowest* completeness anywhere.

Below 1% coverage over Indian cities is not a training signal. Keep OSM as a sparse
*validation* cross-check on individual buildings if you like, not as supervision.

### 3.5 CartoDEM / Cartosat — reject as a label ❌

- Open tier is **30 m posting**. Useless at our scale.
- The 2.5 m Cartosat-1 DSM exists (Feb 2024) but is **priced for non-government
  entities** — open only to Indian government bodies — and its stated vertical accuracy
  is **8 m LE90**, five times our current error.
- Cartosat-3 pan is 0.25 m and is genuinely excellent *imagery* — but Bhoonidhi's open
  data policy puts anything finer than 5 m behind payment for us.

**Where Cartosat does matter: as unlabelled input imagery.** 0.25 m pan is inside our
trained GSD band, and unlabelled Indian imagery is exactly what a self-supervised or
shadow-consistency-style term would consume. That is a different plan from supervision.

### 3.6 Our own drone survey — still the highest-value item

Unchanged from HANDOFF §9 item 6. OpenDroneMap over a campus or a nearby town produces
a genuinely Indian, genuinely high-resolution orthophoto + DSM pair with known
provenance. **Downsample the orthophoto to the trained GSD band before inference**, or
the resulting number is flattering and meaningless.

This is the only route to an *evaluation* set for India. Every source above is either
too coarse to evaluate against or too noisy. **Without one, "we fine-tuned for India" is
an unfalsifiable claim** — and that is the sentence a judging panel will find.

### 3.7 Rejected — why they don't fit

| dataset | why not |
|---|---|
| **M4Heights** (Sci Data 2025) | Estonia / Netherlands / Switzerland; reference heights at **10 m**. Wrong continent, 30× too coarse. |
| ISPRS Vaihingen / Potsdam | German, tiny, but *does* have DSM+nDSM+semantics — usable as a method sanity check, not for India |
| SpaceNet 1–4 | footprints only, no height |
| WSF3D / GHS-BUILT-H | 90 m / 100 m products. Wrong scale entirely. |

---

## 4 · Dataset survey — the DSM question

> "GAMUS didn't have DSM — is there a dataset with GAMUS's features **and** DSM?"

**Yes: US3D / DFC2019. It is the single cleanest answer to this question.**

### 4.1 US3D (Urban Semantic 3D) — DFC2019

Verified against the dataset's own file description:

| suffix | contents | dtype |
|---|---|---|
| `RGB` | 3-band input image | uint8 |
| `MSI` | 8-band VNIR, pan-sharpened | uint16 |
| `CLS` | semantic classification labels | uint8 |
| `AGL` | **above-ground-level height, metres** | float32 |
| `DSM` | **WGS84 Z coordinate, metres — absolute elevation** | float32 |
| `DSP` | stereo disparity | float32 |

That is GAMUS's exact triple (`RGB` / `CLS` / `AGL`) **plus an absolute `DSM`**.

- Cities: Jacksonville FL (53 tiles) and Omaha NE (54 tiles), ~100 km².
- Imagery: WorldView-3, pan GSD **~0.30 m** — near-identical to GAMUS's 0.33 m.
- Truth: airborne LiDAR at 80 cm, rasterised to 0.5 m.
- Tiles 2048² (~600×600 m) for tracks 1–3; Track 1 is exactly our task —
  single-view → semantic + AGL.
- Open access on IEEE DataPort, `10.21227/c6tm-vw12`.

**And the part that matters most: `DTM = DSM − AGL`, per pixel, exactly.** GAMUS ships
no DTM (verified in §3 of the session-2 handoff against the HuggingFace file listing),
so `DSM = DTM + AGL` has never been validated anywhere in this project. US3D lets us
validate it, on real data, for the first time.

### 4.2 Why this matters more than it looks — it de-risks Divyanshu

This is the dataset that tests the **team's riskiest untested assumption.**
Session-2 §3 established: GAMUS has no DTM, SRTM is not bare earth and sits near
rooftop level over cities, so adding our AGL to an SRTM base double-counts buildings.
Nobody has been able to check this because no dataset in the project had both surfaces.

With US3D you can, concretely:

1. Derive `DTM = DSM − AGL` on 107 tiles of real LiDAR.
2. Fetch SRTM over Jacksonville and Omaha, and **measure** how far SRTM sits above true
   bare earth in an urban core. That number is currently a claim; it becomes a
   measurement.
3. Run our A7 model on US3D RGB, add the derived DTM, and score the resulting **DSM**
   against US3D's own DSM. That is an end-to-end test of the full team pipeline —
   Ritvik's box plus Divyanshu's box — which has never once been scored.

**Do this before any India work.** It is a US dataset and it will not help the domain
gap at all, but it converts the pipeline's biggest unvalidated assumption into a number,
using data that is free, open, and already at our GSD. If `DSM = DTM + AGL` has a bug in
it, everything downstream of us is wrong regardless of how good the Indian fine-tune is.

### 4.3 US3D is also TRAINING data — the sensor-shift rehearsal

Filing US3D purely as validation under-sold it. It is *also* the best additional
training source available, for a reason nothing else can offer:

- `RGB` + `CLS` + `AGL` — the **exact same triple** as GAMUS, so it is a drop-in.
- **~0.30 m GSD** against GAMUS's 0.33 m — effectively the same scale.
- **WorldView-3 satellite** against GAMUS's aerial orthophoto — a genuine *sensor*
  shift, which is the dominant component of the India shift too.
- 0.8 m airborne LiDAR truth — real ground truth, not stereo-derived.

That combination is unique here. India's labelled options are noisy (DFC2023's nDSM is
2 m stereo) or weak (Open Buildings, ~4 m effective). **US3D is the only place we can
rehearse a sensor shift and then check whether we got it right.** If the fine-tuning
recipe cannot survive WorldView-3 with clean LiDAR to score against, it will not survive
New Delhi with 2 m stereo labels — we just would not be able to tell.

So US3D carries two roles, and **neither of them is a second model**:

1. extra AGL training data carrying a real sensor shift, mixed into batches exactly as
   India will be;
2. the DSM/DTM validation of §4.2, run on frozen v0.1.0 weights, never touching the
   training path.

### 4.4 Should we add a DSM head?

**No.** Recommend against, and the reason is in HANDOFF §1's team split.

AGL is *the* correct output for us. It is scale-and-offset-free by construction, which
is precisely what makes it robust across cities — our held-out NYC building MAE was
1.585 m even while overall bias was −2.02 m. A DSM head would have to predict absolute
elevation above a geoid, which is (a) not recoverable from a single nadir image without
external terrain, and (b) explicitly Divyanshu's box.

What US3D justifies is **an auxiliary DTM-consistency check at evaluation time**, not a
third head. Keep the contract: we emit AGL, he adds terrain.

---

## 5 · Ordered plan

Each item says what it produces and roughly what it costs.

**There is ONE model throughout.** Domains stack into it by mixed batching; no step
below forks a second set of weights.

```
v0.1.0   GAMUS only                     held-out NYC building MAE 1.585 m   [frozen]
  +      US3D            sensor shift, CLEAN truth, verifiable
  +      DFC2023 Delhi   India, noisy 2 m stereo labels
  +      Open Buildings  weak, ~4 m effective, anchors absolute height only
v0.2.0   one model, GAMUS still in every batch
```

| # | task | output | cost | blocks on |
|---|---|---|---|---|
| 0 | **RGN probe, A7, DC+PHL vs NYC** | Which blocks respond to a real shift we have already measured | 40 min GPU | nothing — data is on the cluster |
| 1 | Register IEEE DataPort; start US3D download | Unblocks 2, 3, 6 | manual + hours of transfer | Ritvik doing the registration |
| 2 | Derive DTM = DSM − AGL; score SRTM against it | First real number on the team's `DSM = DTM + AGL` assumption | ~1 day, no GPU | task 1 |
| 3 | Zero-shot v0.1.0 on US3D Track 1 | A second held-out-domain number, different *sensor*, free | 2 h GPU | task 1 |
| 4 | Layerwise-LR + block-freeze in `param_groups()` | The mechanism the sweep needs | done, see §2.4 | — |
| 5 | F0–F3 block sweep, ViT-S, **2 seeds each** | A ranking that survives the 4.3% noise band | ~8 h GPU | task 0 informs it |
| 6 | Add US3D to training, mixed batches | Sensor-shift adaptation with checkable ground truth | ~6 h GPU | tasks 1, 5 |
| 7 | Download DFC2023 Track 2; **count the New Delhi tiles** | Go/no-go on the only Indian labelled source | 1 day | registration |
| 8 | Add Delhi + Open Buildings, ratio swept → ViT-L | The India model, v0.2.0 | ~6 h GPU | 6, 7 |
| 9 | Drone survey + ODM, downsampled to trained GSD | The only honest Indian evaluation set | days, not GPU | — |

**Task 0 is the only one that needs nothing new — run it first. Task 1 is manual and
gates half the list, so start the registration in parallel and let the download run
overnight.**

Report the GAMUS number after every one of 6, 8. A fine-tune that gains 2 m on India and
costs 0.5 m on GAMUS is a legitimate trade, but it has to be visible to be traded.

### The number to promise

Do not promise a GAMUS-like figure for India. Published deep-learning height models in
Global South cities land at **2.2–7.0 m MAE** (Nairobi / Kathmandu / Quito, Sci Rep
2025). Our GAMUS held-out figure is 1.585 m on buildings. **An honest India target is
3–5 m MAE, with the correlation held above 0.8** — and with the caveat, stated every
time, that until item 8 exists we have no Indian ground truth to verify it against.

---

## 6 · Sources

- Lee et al., *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*,
  ICLR 2023 — arXiv 2210.11466
- *2023 IEEE GRSS Data Fusion Contest* (DFC2023) — grss-ieee.org, ieee-dataport.org
- *Data Fusion Contest 2019 / US3D* — ieee-dataport.org, doi 10.21227/c6tm-vw12;
  file spec from github.com/pubgeo/dfc2019
- *Evaluating the suitability of GlobalBuildingAtlas for digital urban infrastructure
  analysis in India*, IOP Env. Res.: Infrastructure & Sustainability 2025 —
  doi 10.1088/2634-4505/ae7f49
- *Open Buildings 2.5D Temporal* — sites.research.google/gr/open-buildings/temporal
- *Narrowing the gap for city building height predictions*, Scientific Reports 2025 —
  doi 10.1038/s41598-025-15929-2
- Biljecki et al., *Quality of crowdsourced geospatial building information*,
  Building and Environment 2023
- *M4Heights*, Scientific Data 2025 — doi 10.1038/s41597-025-06495-3
- CartoDEM / Cartosat specs — Bhoonidhi, NRSC
- DAv2 configs — local HF cache (Small) and huggingface.co (Large)
