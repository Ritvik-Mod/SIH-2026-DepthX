# DepthWizard — session-2 handoff

Read `HANDOFF.md` FIRST, in full. It is still accurate and is the primary document:
the problem, the team split, the model argument, GAMUS facts, the cluster, and bugs
1–15. This file is the delta — everything that happened in the session after it was
written, on 2–3 Sept 2026.

Nothing in HANDOFF.md has been invalidated. Two of its numbers have been superseded
and are marked below.

---

## 0 · How to work with Ritvik — additions to HANDOFF.md §0

Everything in §0 still holds. On top of it:

**He goes by "Mod"** (his surname). Address him that way.

**Commands are the deliverable.** When he asks for commands: one per fenced `bash`
block, and label every single one **MAC terminal** or **HPC terminal**. He gets
impatient when this is wrong. Give the whole sequence in order when he asks for "all
commands" — he pastes them one after another.

**Do not commit or push.** He removed the `Co-Authored-By: Claude` trailer from all 32
commits with `git filter-branch` and force-pushed. Make file changes, then hand him a
paste-ready command:
`cd "/Users/ritvikmod/SIH 2026" && git add -A && git commit -m "…" && git push origin main`
**Never add a co-author trailer.** He asked for this explicitly and it is his repo.

**Verify before you claim.** This session cost three round trips to guesses that the
logs already answered. He never complains about a tool call; he does notice a wrong
assertion. Read the file, run the arithmetic, then speak.

**Correct yourself plainly and move on.** In this session I was wrong about: OOM as the
holdout crash cause, a per-user queue cap as the second cause, the shadow-loss zero
alarm (I over-corrected on a 10-batch sample when the 50-step log was already on disk),
a stale log I misread as a fresh failure, and — the big one — the architectural
diagnosis of the blur. He responded well every time. Own it in a sentence, give the
corrected number, continue.

**His shell loses the venv on every fresh SSH login.** Prefix HPC commands with:
`source /apps/anaconda3/bin/activate deeplearning && source ~/envs/depthwizard/bin/activate && cd ~/SIH2026 && …`
Also `export HF_HOME=~/SIH2026/data/hf_cache` before anything that builds a model, or it
re-downloads 1.27 GB.

**He works at 2–4 am under deadline pressure.** When he asks "is this fine / what do I
do", give the answer in the first line, then detail. When he says "do not argue", state
the concern once in a sentence and then do the work in full.

---

## 1 · Ablation ladder — final results

All ViT-S / 60 epochs unless noted, official split (NYC in train AND test → in-domain).
`best building MAE` is the checkpoint-selection metric.

| run | building MAE | note |
|---|---|---|
| A0 stock DAv2, global affine | 4.830 | r 0.405 |
| A0 stock DAv2, per-tile ORACLE affine | 3.153 | r 0.817, unreachable ceiling |
| A1 regression head | 1.810 | |
| A2 + adaptive bins | 1.769 | |
| A3 + FiLM on log(GSD) | 1.848 | |
| A4 + semantic head | 1.824 | |
| A5 + domain randomisation | 1.841 | |
| A6 + shadow loss | **1.732** | 40 ep fine-tune from A5 |
| A6 CONTROL (same, w_shadow=0) | **1.739** | 40 ep fine-tune from A5 |
| A7 ViT-L, 40 epochs | **1.549** | best model |

**The whole ViT-S ladder A1–A5 sits in a 1.769–1.848 band — 4.3%, single seed, no
repeats. You cannot rank those configurations from this data. Say so.**

**The shadow loss does not work.** A5 1.841 → control 1.739 → shadow 1.732. The shadow
term buys **0.007 m (0.4%)**. A6's apparent gain was 40 extra epochs at a decayed
learning rate. The loss is mechanically sound — occlusion IoU 0.91–0.93 vs an
independent numpy reimplementation, sign convention verified, gradients flow, term live
at ~0.72 throughout — it simply does not move the metric on GAMUS at weight 0.1
supervising 38% of tiles. **This is a clean negative result on a properly controlled
comparison and should be reported, not buried.**

**A7 is the only unambiguous win.** 1.549 is 12% below the best ViT-S, several times the
noise band, and achieved on 40 epochs against A5's 60 — so the confound runs against it.
Backbone scale beat every architectural addition combined.

### Held-out city (NYC never trained on), ViT-L, 40 epochs

| | in-domain (A7) | held-out NYC |
|---|---|---|
| building MAE | 1.549 | **1.585** (best) / 1.597 final |
| building r | — | **0.901** |
| building bias | — | −0.387 |
| overall MAE | 1.234 | **2.849** |
| overall r | 0.929 | **0.739** |
| overall bias | −0.011 | **−2.019** |
| sharpness | 0.084 | **0.023** |

Buildings transfer almost perfectly (2.3% gap) and building correlation holds at 0.901
vs 0.912 in-domain. **But a −2.02 m systematic offset appears on the new city and
overall correlation drops 0.93 → 0.74.** The offset is exactly what Divyanshu's
scale+offset calibration removes — that part of the domain gap is his and it is fixable.
The correlation drop is structure, and per §1 that one is Ritvik's.

Caveats to state every time: measured on the 400-tile `holdout_val` set, which
`train.py` also checkpoint-selects on, so mildly optimistic. NYC is a *different urban
fabric*, not a taller one (max 99.7 m).

### Still not run
- `dw_holdout` (ViT-S held-out) — crashed at epoch 56/60, resumable via `resume=auto`
- `dw_bench` — inference timing, was held, never released
- `--w-grad 6/20/60` refiner sweep — command written, never executed

---

## 2 · Bugs 16–22, found this session

Same pattern as 1–15: silent divergence between intent and behaviour, almost none
surfacing as an error.

**16. A6 was never a fine-tune.** `a6_shadow.yaml` said "fine-tune from the A5
checkpoint" and nothing implemented it. `resume=auto` only reads `out_dir/last.pt`,
which was empty, so A6 would have trained from the pretrained backbone for 40 epochs
(A1–A5 get 60) at learning rates 3.3× below base. It would have finished cleanly,
undertrained, and read as "the shadow loss hurts". Fixed by adding `train.init_from` to
`train.py` — weights only, epoch 0, fresh optimiser, skipped when a real resume
happened, raises on a missing path.

**17. `binary_cross_entropy` is blocked under bf16 autocast.** A6 warm-started
correctly and died on the first backward. `losses/shadow.py` now forces that op to fp32
via `torch.autocast(enabled=False)`. Deliberately NOT switched to
`binary_cross_entropy_with_logits`: `occ` carries a −1e4 sentinel, so the unclamped
logit form changes the loss magnitude on confidently-unshadowed pixels. Verified
identical: 0.6738 both ways. `tests/test_shadow.py` case 5 now runs the loss under
autocast on cpu and cuda. **All five test modules passed throughout because every test
ran fp32 — the shadow loss had never once executed in the precision the trainer uses.**

**18. City-holdout eval sets were keyed by directories that do not exist.**
`build_splits` returned `{"holdout_val": …}` / `{"holdout_test": …}`, but `ConcatSplits`
builds one `GamusDataset` per key and `_open` reads `root/images/<split>/`. Training was
keyed correctly, so both holdout jobs trained a full epoch and died at the first
validation with `FileNotFoundError`. **The held-out number had therefore never once been
computed in this project.** Fixed by grouping eval tiles under their real split names.
`test_data.py` passed throughout because it asserted city disjointness on the returned
*lists* and never built a dataset from the returned *keys* — a test that was true and
useless at the same time. It now opens a tile through each mapping.

**19. File-descriptor exhaustion.** Both holdout runs died at *identical* step 18837
(epoch 23). `train.py:40` sets `persistent_workers=True`, so workers live for the whole
run, and `gamus.py` cached every h5py handle and never closed one — ~19,700 open files
per worker across 6,557 tiles. The official split's 5,004 tiles sat just under the
ceiling, which is why only city-holdout crashed. Two fixes: `file_system` sharing
strategy in `train.py`, and `ulimit -n` raised to the hard limit in `_common.sh`. That
moved the crash from epoch 23 to 56; the real fix is the LRU bound on the handle cache
(`MAX_OPEN = 192`, commit `7bc822c`) — **not yet validated by a full run.**

**20. Demo tile selection returned the emptiest tiles.** "Best 12 by building MAE"
picked twelve PHL tiles with true p50 of 0.00–0.13 m and building r as low as 0.511
against the model's 0.908 — a flat field has nothing to get wrong. Fixed with a content
floor: `--min-building-frac 0.10 --min-p99 15`.

**21. Sun-angle cache is fitted at a different march length than the loss consumes it.**
`estimate_sun.py` uses `max_dist_px=150`; `configs/base.yaml:65` gives the loss 200.
Measured impact: 0–3° elevation shift against a 36–42° half-max width. **Left alone
deliberately** — too small to matter, and 200 is the better physical cap.

**22. `max_sun_elev: 60` silently drops 299 cached tiles**, 193 of them PHL (26.8% of
that city). **A6's real supervision is 2,227 of 5,863 train+val tiles = 38%, not the 43%
in HANDOFF.md §8.** Use 38%.

---

## 3 · Sun angles — investigation closed, A6 was justified

HANDOFF.md §8 asked whether the per-city azimuth spread justified running A6. It does.

Per-city pooled spread looked bad (DC ±22.6, NYC ±35.0, PHL ±35.9) but the
distributions are **bimodal — two sorties per city**. Two-cluster circular split gives
within-cluster spread of **9.4–18.9°** for five of six clusters, at the search's own 10°
resolution floor.

Tested and refuted: antipodal degeneracy (`±45 of anti` = 0.000/0.002/0.004) and a
**mirror degeneracy about north–south** (`az` vs `360−az`, which the antipodal test does
not cover — occlusion marches along `u = (sin az, −cos az)`, so `az → 360−az` mirrors
about N–S and rectilinear street grids are near-symmetric about that axis). Direct
rescore: mirror never wins, median margin 0.054–0.089 on peaks of 0.17–0.22.

Azimuth half-max width is **±50–60°**, so the ±9–19° estimation error is comfortably
inside tolerance. Elevation is real but coarse (half-max 36–42°), giving ~1.5× spread in
the `tan θ` height scale — a perturbation at w_shadow 0.1, not a poison.

Scripts: `scripts/check_sun_degeneracy.py`, `hpc/22_sundegen.pbs`. Outputs on the
cluster: `data/GAMUS/sun_iou_150.json`, `sun_iou_200.json` (per-tile IoU + mirror
margin, which `estimate_sun.py` computes and discards).

**Cross-check worth keeping:** the HF release has RGB 6557 + IMG 2167 = 8724, so
NYC = 2,167 tiles and DC+PHL = 6,557 — exactly the `train tiles=6557` the holdout run
reported. Independent confirmation the holdout split is correct.

**GAMUS ships NO DTM.** Verified against the HuggingFace file listing: 26,174 files =
8,724 × 3 plus README and .gitattributes; suffixes are only CLS/AGL/RGB/IMG. So
`DSM = DTM + AGL` cannot be validated on this dataset. Divyanshu's terrain must come
from outside, and SRTM is not bare earth — over cities it sits near rooftop level, so
adding AGL to it double-counts buildings. **For a single 338 m tile, a flat base plane
is more correct than an SRTM base.**

---

## 4 · The sharpness problem — the current open thread

GAMUS truth has sharp building edges; the prediction is smooth. Sharpness ratio
(`metrics.py:73`, gradient at TRUE-edge locations — mean gradient does not measure this,
that was bug #5) sits at 0.084–0.133 depending on configuration. 1.0 would match truth.

**It has not moved under anything tried:**

| approach | sharpness | building MAE |
|---|---|---|
| baseline (single 518 crop) | 0.129 | 1.667 |
| guided filter, best of 9 settings | 0.123 | 1.664 |
| semantic flattening, best of 6 | 0.108 | 2.458 |
| learned full-res refiner, 6 epochs | 0.133 | 1.669 |
| deployed tiled path | 0.125–0.133 | 1.502 |

**Calibration (`scripts/sharpness_ceiling.py`, on synthetic — RUN IT ON REAL GAMUS,
that is outstanding):** degrading the ground truth and scoring it against itself gives
1 px shift → 0.500, blur σ=1 → 0.398, σ=2 → 0.202, **σ=3 → 0.135**. So the model's
output is equivalent to the truth blurred at **σ ≈ 3 px ≈ 2 m**, and the metric is not
saturated — 0.13 is genuinely poor with real headroom.

**Why each failed, mechanistically:**
- *Guided filter*: `--guide sem` uses the semantic head's probability, and `heads.py:125`
  gives `SemanticHead` the same 296²→518² bilinear upsample. The guide is as blurry as
  the target, so the filter is a smoother. `--guide gray` (RGB, genuinely sharp at full
  res) was never tested.
- *Semantic flattening*: `ndi.label` returned 851 instances over 60 tiles ≈ 14/tile,
  where a 171 m urban tile holds far more. Touching footprints merge, one median gets
  assigned to buildings of different heights, and true edges between them are **erased**
  — which is why sharpness went *down*. `--oracle-sem` (IoU 0.950) was **worse** than
  predicted (0.763), ruling out the semantic head as the bottleneck. Fix would be
  persistence-thresholded watershed (`skimage.morphology.h_maxima` is exactly H0
  persistence simplification) — not yet tried.
- *Refiner*: 67k params at full resolution with sharp RGB available, and sharpness moved
  0.129 → 0.133. **This is evidence against my own diagnosis** (below).

**Current best hypothesis, NOT yet tested — this is where to go next.**
`heads.py:110`:
```python
h = (p * centres[:, :, None, None]).sum(dim=1, keepdim=True)
```
That is the **conditional mean** of the bin distribution. HANDOFF §3's whole argument is
that at a roof edge the truth is bimodal (0 m or 18 m) and regression is forced to emit
~9 m in mid-air — but the expectation of a bimodal distribution *is* 9 m. The bins head
represents the bimodality correctly and discards it in the last line. This explains why
A2's bins changed nothing, why sharpness is identical across A1/A2/A6, and why no
post-process recovers it: the information is destroyed inside the head.

**The cheapest test needs no retraining.** A7 uses the bins head and already emits
`bin_logits`. Decode with the **mode** (`centres[argmax(p)]`) or a temperature-sharpened
softmax instead of the mean. Same weights, different last line. If sharpness jumps from
0.13 toward 0.3–0.4 with building MAE intact, that is the answer and it is a one-line
change. **Write this script first if he returns to sharpness.**

Ranked after that: entropy penalty on `p` during training; an edge-conditioned gradient
loss (the current `multiscale_gradient_loss` avg-pools 3 of its 4 scales, destroying the
fine detail it is meant to score); learned upsampling to replace the 296²→518² bilinear
jump; the `--w-grad 6/20/60` sweep that was queued and never run.

**A diagnosis I got wrong, stated so the next agent does not repeat it.** I argued the
blur was caused by the heads emitting at 296² and bilinearly upsampling to 518², and
predicted a full-resolution refiner would break that ceiling. It did not. σ=3 px at 518²
is σ≈1.7 px at 296², so most of the blur exists *before* the upsample. The upsample is
real but is not the binding constraint.

**TDA cannot sharpen edges.** Persistence measures the *prominence* of features
(amplitude between birth and death), and a sharp step and a σ=3-blurred step have
essentially identical persistence diagrams. It is invariant to exactly the quantity being
measured. TDA *can* do three useful things: persistence-thresholded watershed for
building instance separation (fixes the flattening failure), cleanup of ringing after
aggressive sharpening, and a diagnostic — the bottleneck distance between prediction and
truth diagrams measures **how many small structures the blur has erased**, which MAE is
blind to. Instance separation is a clean ask to hand Vineeth, whose box TDA is.

---

## 5 · Inference path — measured, and better than expected

`predict_scene` (coarse levelling pass, 518² tiles at 25% overlap, ramp-blended) versus a
single 518² forward, **scored on the identical centre 518² window**:

| | overall MAE | building MAE | building r | sharpness |
|---|---|---|---|---|
| single 518 crop | 1.891 | 1.588 | 0.904 | 0.130 |
| tiled ov=0.25 level=on | **1.789** | **1.502** | **0.912** | 0.133 |

**Overlapped tiling improves building MAE by 5.4%** — an ensembling effect, several
predictions averaged per pixel. It does not blur. `overlap: 0.25` is a good default;
0.5 buys almost nothing.

**Border degradation, worth acting on:** centre 518² gives building MAE 1.502, the full
1024² tile gives 1.618 — ~8% worse toward the scene edge, consistent with border pixels
getting less overlap coverage and less context (a shadow falling outside the frame
cannot be used). **Mirror-padding by half a tile in `infer.py` before tiling would fix
this.** Not done. Tell Divyanshu the outer ~130 px of any exported scene is less reliable.

Beware the confound I hit: `single crop` scores the centre 518² while the full tile
scores 1024², so band statistics across them are NOT comparable. `sharpen_0_pathcheck.py`
now prints MATCHED and UNMATCHED blocks separately.

---

## 6 · New scripts (all UNTRACKED — `git status` shows `??`)

Committed work is in the log; these nine are written, tested, working, and **not
committed**. He may or may not want them in the repo.

| file | purpose |
|---|---|
| `scripts/compare_predictions.py` | predict-vs-truth 4-panel renders; `--scan/--best/--random`, content floor, `--rank composite`. Truth and prediction share ONE colour scale (truth's p99) — independent scales flatter any model. Selection reason is stamped on the figure and in the filename. |
| `scripts/sample_preview.py` | RGB / height / land-cover panels for showing the team the dataset |
| `scripts/export_for_threejs.py` | one AGL+RGB → `heightmap.tif` (float32 real metres), `texture.png`, `heightmap_preview.png`, `metadata.json` |
| `scripts/export_batch_threejs.py` | batch version, `--from-json <compare.json>`, `--source pred\|gt`, one model load |
| `scripts/predict_external.py` | arbitrary external imagery; **resamples to the trained GSD band** before inference |
| `scripts/bench_inference.py` | real inference timing (the `04_probe.pbs` figures are TRAINING throughput and understate inference ~3×) |
| `scripts/sharpness_ceiling.py` | calibrates the sharpness metric by degrading the truth |
| `scripts/sharpen_{0,1,2,3}_*.py` + `_sharpen_common.py` | path check, guided filter, semantic flattening, learned refiner — shared metric harness |
| `configs/ablations/a6_control.yaml`, `holdout_vitl.yaml` | the shadow control, and ViT-L held-out |
| `hpc/22_sundegen.pbs`, `hpc/23_demo12.pbs` | sun diagnostic, demo-set generation |

---

## 7 · Deliverables produced

- `outputs/handoff/` — SPEC.md three-file bundle for one tile
- `outputs/demo12/`, `demo12_test/` (24 images: 12 bMAE-ranked + 12 composite-ranked),
  `demo12_train/` — comparison panels
- `outputs/threejs_export/` (12 train tiles, predictions), `threejs_export_gt/` (same 12,
  ground truth — for isolating whether a render problem is the model or the pipeline)
- `outputs/india_export/`, `campus_export/`, `iisc_export/` — external imagery
- `outputs/refiner/refiner.pt`

**Numbers that survive scrutiny, for the report:**
- Best 12 of 428 content-rich **test** tiles (never seen): mean building MAE **0.739 m**
- Content-rich median across those 428: **1.600 m**; val independently gives **1.560 m**
  — two unseen splits agreeing to 2.5%
- Worst tile in the test pool: building MAE **46.6 m**. Know this before you are asked.
- Train-split pool median 1.339 vs test 1.600 — a **19% memorisation gap**. This is the
  honest overfitting number. The top-12 comparison (0.777 vs 1.185) is **inflated** by
  pool size (2104 vs 428 eligible) and should not be reported.

**Train-split outputs exist and are for pipeline development only.** He asked for the
best 12 from train, for Vineeth's persistence and Aakarsh's rendering to build against —
those are the cleanest fields available and provenance does not matter for that use.
Every file carries `"split": "train"` and `field_source`. **Keep it that way.** Do not
let train-derived numbers near an accuracy claim; the first question anyone asks about a
good result is whether the tile was in training.

---

## 8 · GitHub

`github.com/Ritvik-Mod/SIH-2026-DepthX`, **private**, collaborators Divyanshu4501,
CallMeChandler and Vineeth-Varanasi. Kept private deliberately: `HANDOFF.md:203` and
`hpc/RUNBOOK.md:9` contain the cluster username and the college's public SSH IP.
Scrubbing them was considered on 3 Sept 2026 and **deliberately declined** — the repo is
private, the pair is a username plus a public endpoint with no password, and a
`filter-branch` rewrite would have forced every collaborator to re-clone mid-sprint.
**Revisit before it is ever made public; git history retains them.**

---

## 9 · What to do next, in priority order

1. **Mode-vs-mean decode test** — no retraining, minutes to run, and it is the leading
   explanation for a problem that has resisted four approaches.
2. **Run `sharpness_ceiling.py` on real GAMUS** — the σ≈3 calibration is from synthetic
   slabs; real LiDAR edges have different statistics.
3. **Re-queue `dw_holdout`** (ViT-S held-out) — crashed at epoch 56/60, `resume=auto`
   continues it, and the LRU handle-cache fix is untested by a full run.
4. **Mirror-pad in `infer.py`** — measured 8% border degradation in the shipped path.
5. Report the shadow-loss negative result and the A1–A5 noise band honestly.
6. Still the highest-value item overall, unchanged from HANDOFF.md §9: **nothing has
   been validated on Indian data.** The external-image runs are qualitative only — no
   ground truth, assumed GSD, one image possibly AI-generated, one obliquely angled.
