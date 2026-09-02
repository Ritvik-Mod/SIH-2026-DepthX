# DepthWizard — full handoff

You are picking up an in-flight project. Read this whole file before acting.

---

## 0 · How to work with Ritvik

B.Tech CSE, 2nd year, BIT Mesra (CGPA 8.9). Background is **medical imaging AI** —
he hand-built a 9-organ H&N segmentation model matching nnU-Net, and is separately
rebuilding nnU-Net's planner from scratch. So: he knows CNNs, encoder–decoders, U-Net,
training loops, Dice/loss design, and ablation discipline. He was **new to remote
sensing and to ViTs/attention** at the start of this project and has been learning them
here.

**Register.** Plain and specific. Numbers over adjectives. He works in a measured,
ablation-driven style — locked test sets, noise floors, seed controls — and has
published his own headline downward when it turned out to be an artifact. Match that.

**He actively wants correction.** Flag confounds, noise floors and confers unprompted.
Never let him overclaim; he will thank you for the correction, not resent it. If a
number looks too good, say so and find the leak.

**Read the ask.** When he asks a conceptual question, teach properly — layman first,
then the picture, then the technical level, and explain every term rather than assuming.
When he asks for commands, give **commands only**: one per fenced `bash` block, and
always say explicitly whether each runs on the **MAC terminal** or the **HPC terminal**.
He gets (reasonably) impatient when he asks for the latter and receives the former.

**Verify, don't assume.** Nearly every serious bug in this project was found by reading
a file or doing arithmetic on an output, not by a test passing. Several were found
because *he* pasted output that contradicted an assumption. Keep that habit.

---

## 1 · The problem

**SIH26175 — DepthWizard (ISRO).** Convert a single nadir satellite/drone image into a
navigable 3D terrain in the browser.

Official problem repo: `github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026`
- Specifies the **GAMUS** dataset and SRTM as a supplementary DEM
- Evaluation: **50% DSM accuracy** (RMSE, MAE, **correlation** vs LiDAR, *across
  landscapes*) + **50% rendering & UX**
- (Could not be independently confirmed as officially SAC/ISRO; content is specific and
  consistent. Treat as authoritative but verifiable.)

**Team split** — Ritvik owns only the first box:

| Person | Owns |
|---|---|
| **Ritvik** | image → per-pixel height above local ground (nDSM/AGL), in metres |
| Divyanshu (lead) | metric calibration + add terrain (DTM) → DSM |
| Vineeth | TDA (persistent homology) used to detect and smooth noisy regions |
| Aakarsh | Three.js flythrough |

**Decided:** the India domain gap is **Ritvik's** problem, not Divyanshu's. Calibration
is a scale+offset fit; it can rescale a wrong-shaped field but cannot repair structure.
Ritvik owns "the shape is right", Divyanshu owns "the units are right". This maps onto
the SILog/L1 split in the losses and onto correlation-vs-MAE in the metrics.

**Downstream constraints that shape the output:** Vineeth runs persistence on the field,
so it must be **float32** (8-bit quantisation creates tied values → degenerate critical
points), **unsmoothed** (smoothing is his job), with explicit nodata, and **globally
consistent** (tile seams read as high-persistence structure).

---

## 2 · The repository

`/Users/ritvikmod/SIH 2026`, git on `main`, ~2,900 lines of Python. Local venv `.venv`
(python 3.13, torch 2.13, Apple MPS) for development only.

```
heightmap/
  config.py            YAML + dotlist overrides, with guards that reject bad combos
  data/classes.py      GAMUS class ids — the authority, cites the paper
  data/gamus.py        dataset, splits, h5 handling
  data/transforms.py   augmentation, scale→GSD supervision
  models/net.py        HeightNet: backbone + neck + swapped heads
  models/heads.py      bins / regression / FiLM / semantic / uncertainty
  losses/terms.py      SILog, L1, multi-scale gradient, bin CE, chamfer, semantic, NLL
  losses/shadow.py     shadow-consistency loss
  losses/combined.py   weighted assembly, logs every term separately
  metrics.py           metrics + exact streaming accumulator + sharpness
  train.py             loop, param groups, warmup/cosine, EMA, atomic ckpt, resume
  infer.py             tiled inference with coarse-guided levelling
  export.py            3-band float32 GeoTIFF + JSON sidecar + preview
configs/base.yaml, configs/ablations/a1..a7 + holdout.yaml
scripts/  sanity.py dataset_stats.py zero_shot_eval.py estimate_sun.py
          repack_gamus.py make_synthetic.py collect_results.py download_gamus.py
tests/    test_data test_model test_metrics test_infer test_shadow
hpc/      RUNBOOK.md + PBS job scripts
```

`scripts/make_synthetic.py` generates GAMUS-shaped fake tiles (with physically-cast
shadows) so the whole pipeline is testable without the 80 GB download. Use it.

---

## 3 · The model, and why

**DINOv2 ViT encoder + DPT neck**, both inherited from Depth Anything V2 via
`transformers.AutoModelForDepthEstimation`. The depth head is **discarded**.

The argument, which Ritvik can defend and you should not weaken: DAv2's training loss is
**affine-invariant by construction** — `a·d + b` scores identically to `d`. Absolute
scale was not lost, it was *optimised away*. It cannot be recovered by rescaling the
output, so the head is replaced rather than calibrated. **This is now measured on their
own data** (see A0 in §6): with a per-tile *oracle* affine DAv2 reaches r = 0.82, with a
single global affine it collapses to r = 0.34.

Second argument, for the backbone choice: the dominant physical cue for height in nadir
imagery is **shadow**, which is long-range (a 30 m building at 0.3 m GSD with a 30° sun
casts a ~170 px shadow). A 3×3 CNN needs ~85 layers to connect roof to shadow tip;
attention does it in one hop.

**Shapes:** 518² input, patch 14 → 37×37 = 1369 tokens (+1 CLS). `neck[-1]` is
`(B, 64, 296, 296)`. Heads operate there and upsample to 518².

**Heads:**
- **Adaptive bins** (AdaBins / HTC-DC family) — predicts per-image bin centres plus a
  per-pixel distribution; height = Σ p·c. Rationale: at a roof edge the truth is bimodal
  (0 m or 18 m); regression is forced to emit ~9 m, a value in mid-air, which *is* the
  melted-edge failure. A distribution stays bimodal.
- **FiLM on log(GSD)** — satellite metadata gives metres/pixel, so metric output is
  recoverable. Sanity check 4 *proves* that with a constant GSD, FiLM's first layer
  receives exactly zero gradient — which is why scale augmentation exists.
- **Auxiliary semantic head** — regularises shared features and yields the building mask
  needed for building-only metrics.
- **Uncertainty head** (optional) — Gaussian NLL. Introduce only *after* L1 training is
  working; from scratch the model inflates σ to escape the accuracy term.

**Losses** — SILog (α=10, λ=0.85, on `log(h+1)` because AGL ground is exactly 0 and
depth data never has this problem), L1 (the only term anchoring absolute metres),
multi-scale gradient (4 scales; pointwise losses cannot distinguish a sharp wall from a
10-px ramp because the errors cancel), bin CE + chamfer, semantic CE, shadow
consistency. Log every term separately — you cannot debug a sum.

**Shadow-consistency loss** (`losses/shadow.py`) — the differentiator. Renders the
shadow the predicted height field would cast under a given sun angle and compares to
shadows detected in the image.
- Toward-sun unit vector in image coords: `u = (sin az, −cos az)`; verified with
  az=180° (sun due south) → `u = (0,+1)` → shadows fall north. **Do not "fix" this.**
- `occlusion(p) = max_t ( H[p+t·u] − H[p] − t·gsd·tan θ )`, log-spaced t, 16 steps.
  Compares H at both ends, so shadows landing on other buildings are handled correctly.
- Detection uses three conjoined cues: absolute darkness × local-contrast darkness ×
  **relative blueness** (shadows are sky-lit; dark tree canopy is greener, not bluer —
  this alone removed 100% of tree false positives in testing).
- **Property that makes it compose with L1:** adding a constant to H leaves
  `H[p+t·u] − H[p]` unchanged, so it constrains *relative* height only and says nothing
  about absolute scale. It can never train a metric model alone.
- Exclusion mask must come from **ground-truth** classes, never predicted ones —
  predicting "tree" everywhere would zero the loss out, a degenerate escape hatch.
- Validated: occlusion IoU 0.91–0.93 vs an independent numpy reimplementation, azimuth
  recovered by sweep, 180° sign check decisive on every tile, gradients flow.

---

## 4 · GAMUS — verified facts

Source: paper `arXiv:2305.14914`, plus measurements on the actual download.

- **GSD 0.33 m** (paper §1, quoted directly).
- **Classes 1–6:** 1 ground, 2 low-vegetation, **3 building**, 4 water, 5 road, 6 tree.
  `0` is unlabelled. `255` is a nodata value **present only in NYC**.
  Authority is `heightmap/data/classes.py`. Building is **3, not 2**.
- HuggingFace release `earthflow/GAMUS` is **8,724 tiles** (train 5,004 / val 859 /
  test 2,861), three cities **DC, NYC, PHL**, 80 GB, 1024² tiles, key `'image'`.
  The paper claims 11,507 tiles across five cities — the HF release is a **subset**.
- **NYC names its images `*_IMG.h5`; DC and PHL use `*_RGB.h5`.** Heights (`_AGL`) and
  classes (`_CLS`) are consistent.

**Measured statistics (train+val, all three cities):**
```
p50 0.30 m   p90 13.4   p99 32.0   p99.9 44.3   max 392.6 (a DC tile)
frac <= 0.5 m 0.516     frac >= 30 m 0.0133
negatives 2.79% of pixels     non-finite 24,235
class 0 3.78%   class 255 0.07%   class 3 (building) ~17%
per city:  DC mean 6.86 (max 392.6)   NYC mean 4.39 (max 99.7)   PHL mean 2.73 (max 276)
```
**NYC is not the tall city** — max 99.7 m. As the held-out city it tests a *different
urban fabric*, not a harder height range. Say it that way.

**Config decisions settled by these numbers:** `gsd_base: 0.33`, `h_max: 250` (p99.9 is
44 m; raising it to 400 would spread 128 bins over a 99.9%-empty range), `ignore_index:
0`, `n_classes: 7`, crop 518, scale 0.5–1.97 → **trained GSD coverage 0.165–0.65 m/px**
(1 m Cartosat MX is outside it — state that as a limitation).

**Splits:** the official split has **NYC in both train and test**, so it measures
within-city performance. `split_mode: city_holdout` trains on DC+PHL and evaluates on
NYC. **Report both**, and lead with the held-out number.

---

## 5 · HPC — BIT Mesra HPCF

**PBS Pro, not Slurm.** `qsub` / `qstat` / `qdel` / `qhold` / `qrls`.

```
user            btech10848.24
on campus       ssh btech10848.24@172.16.220.100      (RFC1918, campus/VPN only)
off campus      ssh btech10848.24@115.240.90.140
```
The two do **not** hairpin — each is unreachable from the other side. Check before
debugging anything else:
`nc -z -G 5 172.16.220.100 22 && echo campus || echo external`

**Node `rachel-gpu`:** 2× NVIDIA **L40** 46 GB (Ada, cc 8.9), 48 cores, 527 GB RAM,
driver 555.42 / CUDA 12.5, bf16 native. Ten CPU-only nodes on `workq`.

**Environment** `~/envs/depthwizard` (standalone venv): python 3.11.13,
**torch 2.6.0+cu124**, cudnn 90100, transformers 5.16.1, opencv **5.0.0**, numpy 2.4.6.
Data at `~/SIH2026/data/GAMUS`, HF cache at `~/SIH2026/data/hf_cache`.

**Cluster gotchas — all learned the hard way:**
- PBS **does not fence GPUs**. A job granted `ngpus=1` still sees both cards, and other
  users share the node. `hpc/_common.sh` picks the least-used GPU at job start. The
  supplied `gpujobsubmit.sh` template has `ngpus=0`, which yields a GPU-queue job with
  no GPU.
- **Compute nodes have no internet.** All HuggingFace assets are pre-staged from the
  login node; jobs run with `HF_HUB_OFFLINE=1`.
- **PBS does not stream job output** — the `-o` file appears only at job end. Jobs mirror
  to `logs/<jobname>.<jobid>.live` on the shared filesystem; tail that.
- `/home` is 106 TB Lustre with **no quota**. `/tmp` has only ~12 GB free — never use it.
- No `/usr/bin/time`, no CUDA modules (the pip wheel carries its own runtime).
- Jobs start in `$HOME`; every script does `cd $PBS_O_WORKDIR`.
- Training checkpoints **atomically** and **resumes** (`train.resume=auto`) because
  shared queues kill at walltime. Re-`qsub` the identical command to continue; it no-ops
  once the epoch count is reached.

**rsync from the Mac — the excludes must be anchored:**
```
rsync -rtv --delete --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
  --exclude '/data' --exclude '/outputs' --exclude '/logs' \
  ./ btech10848.24@172.16.220.100:~/SIH2026/
```

---

## 6 · Measured results so far

**A0 — stock DAv2-Base zero-shot, 400 val tiles** (`outputs/a0_zeroshot.json`):

| fit | building MAE | building r |
|---|---|---|
| single global affine | 4.830 m | 0.405 |
| per-tile **oracle** affine (unreachable) | 3.153 m | 0.817 |

Per-band bias on the global fit: **−9.9 m** at 10–30 m, **−26.6 m** above 30 m. It
flattens everything tall. This is the ceiling for any post-hoc rescaling of stock DAv2.

**A1 after a single epoch** (ViT-S, regression head): overall MAE 2.239 / r 0.780;
building MAE 3.499 / r 0.479; **30 m+ band bias −17.8, r −0.204**; sharpness ratio 0.050.
Already beats A0 on building MAE. The 30 m+ row and the sharpness ratio are the two
things to watch as the ladder progresses — they are what bins and the gradient loss are
respectively for.

**Throughput on one L40, batch 8, bf16** (from `hpc/04_probe.pbs`):
```
ViT-S 37.4 img/s  6.6 GB     ViT-B 19.7 img/s 10.6 GB     ViT-L 8.0 img/s 23.3 GB
ViT-L at bs=16 OOMs.  Throughput is flat across batch size -> bs 8 everywhere.
A1 (regression head, fewer losses): 125 img/s, 40 s/epoch.
```
The bins head + full loss suite costs a **3.4× slowdown** vs the regression head. The
overhead is the heads and losses, not the backbone. Dataloading is *not* a bottleneck at
125 img/s — **the repack (`scripts/repack_gamus.py`) is unnecessary; do not run it.**

Ladder estimate: A1 ~27 min, A2–A6 ~1.5 h each → **~8 h**, then ViT-B/ViT-L on top.

---

## 7 · Bugs found — and the pattern

Fifteen so far. The pattern worth internalising: **silent drops and unanchored patterns**.
Almost none surfaced as an error; most were caught by arithmetic on an output.

1. `h5py` handles can't be pickled → `__getstate__` drops the cache
2. MPS can't do non-divisible adaptive pooling (518→296) → bilinear
3. Bins head initialised predictions at `h_max/2` = 125 m → analytic exponential init
4. Tile-id collision in the synthetic generator (ids reused across splits)
5. **Sharpness metric couldn't detect blur** — total variation is preserved under
   blurring; must measure gradient *at true-edge locations*
6. **Building class was 2, should be 3** — building-only metrics were measuring
   low-vegetation, and the shadow exclusion mask was inverted (excluding buildings,
   keeping trees)
7. Shadow exclusion used *predicted* semantics — a degenerate escape hatch
8. `rsync --exclude 'data'` matched `heightmap/data/` → `ModuleNotFoundError` on the node
9. `.gitignore 'data/'` — same bug; the package had **never been committed**
10. **NYC's `_IMG.h5` suffix** → 1,167 tiles *and the entire held-out city* silently
    missing. Caught by arithmetic: 187,840,000 sampled px ÷ 40,000 = 4,696 tiles vs
    5,863 expected
11. `estimate_sun.py` hardcoded `_RGB.h5` → would have crashed once NYC was enabled
12. **Class 255** (NYC only) → `cross_entropy` device-side assert, but only on batches
    containing such a tile, so a 4-tile overfit test walked straight past it
13. `/usr/bin/time` not installed on the node
14. Live logs keyed only on job name → two jobs interleaved into one file
15. `bin_ce_loss` materialised a `(B,128,296,296)` tensor to `argmin` over — replaced
    with `searchsorted`, bit-identical, no allocation (perf, not a bug)

**Test state:** 5/5 test modules pass; `scripts/sanity.py` is **16/16 on real GAMUS**
(needs `--overfit-steps 200`; 40 is too few to converge and gives false failures).

---

## 8 · Current state

The ablation ladder was just launched: `bash hpc/31_ladder.sh` chains
**A1 → A2 → A3 → A4 → A5 → A6 → A7 → holdout** as PBS `afterany` dependencies, one GPU
at a time. A1–A6 are **ViT-S / 40 epochs** deliberately — those runs exist to *rank
configurations*, and ViT-S ranks them at 1.9× the throughput; only the winner is scaled
to ViT-B and ViT-L.

Sun angles are estimated and cached: **2,526 of 5,863 tiles (43%)**, elevation p50 40°,
azimuth p50 160° — physically plausible for a leaf-off orthophoto survey at 39–41°N. The
rest were skipped as flat or low-agreement, which is expected. These are **estimates
recovered from shadow agreement, not recorded metadata**, at ~6° elevation / 10° azimuth
grid resolution. Say so in the writeup; do not present them as acquisition parameters.

### What Ritvik is sending you right now

He has just run a per-city sun-angle clustering diagnostic:

```
python -c "
import json,collections,statistics as s
d=json.load(open('$HOME/SIH2026/data/GAMUS/sun_angles.json'))
by=collections.defaultdict(list)
for k,v in d.items(): by[k.split('_')[0]].append(v)
print(f'{len(d)} tiles estimated of 5863 ({100*len(d)/5863:.0f}%)')
for c,v in sorted(by.items()):
    az=[x[1] for x in v]; el=[x[0] for x in v]
    print(f'  {c:4s} n={len(v):5d}  elev {s.median(el):5.1f} +-{s.pstdev(el):4.1f}   azim {s.median(az):6.1f} +-{s.pstdev(az):5.1f}')
"
```

**How to read it.** Tiles from one city come from a handful of flight lines, so their
azimuths should **cluster tightly**. A per-city azimuth spread of **±10–30°** means the
search recovered real signal and A6 is worth running. A spread near **±100°** means it is
fitting noise, and A6 would be training against garbage — in which case say so plainly
and recommend either dropping A6 or reporting it as a negative result. A6 coming out
*worse* than A5 is itself a legitimate finding worth reporting, not something to hide.

He may also send `qstat -u $USER -a` showing the ladder queued.

---

## 9 · What comes next

1. Read the ladder results as they land (`logs/dw_*.live`, `outputs/<run>/log.jsonl`).
   Watch **building-only signed bias** (the long-tail detector) and **sharpness ratio**.
2. `scripts/collect_results.py` builds the results table; every row on **both** the
   official and city-held-out splits — the gap between them is itself a finding.
3. Scale the winning config to ViT-B and ViT-L.
4. Tiled full-scene inference + export, then hand `outputs/<run>/` to Divyanshu. The
   sidecar's `quantity: "AGL"` field is critical — he must **add** terrain, not subtract.
   Warn him that SRTM is not bare earth (it sits near rooftop level over cities) and that
   vertical datums must match or he picks up a tens-of-metres constant offset.
5. India adaptation (pipeline step 13): domain randomisation is already in A5; weak
   labels from GlobalBuildingAtlas + OSM `building:levels`; shadow loss on unlabelled
   Indian imagery. Keep GAMUS in every batch or the model forgets.
6. Drone survey with OpenDroneMap for genuinely Indian validation — **downsample the
   orthophoto to a realistic satellite GSD before inference**, or the number is flattering
   and meaningless.

**Honest accuracy target: 2–4 m MAE.** Not a LiDAR replacement. Lead with the
held-out-city number, not the in-domain one.

## Reference

Four-part handbook written for Ritvik during this project (foundations → architecture →
data/losses/training → inference/eval/handoff):

`hpc/RUNBOOK.md` has the cluster procedure. `SPEC.md` has the output contract.
