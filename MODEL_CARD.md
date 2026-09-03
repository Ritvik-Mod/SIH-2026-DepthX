# Model card — DepthWizard height model v0.1.0 (`a7_vitl`)

Frozen 4 Sept 2026. This is the release the Three.js bundles and every reported number
came from. Later fine-tuning runs (India adaptation) start from here and must not
overwrite it.

## What it does

One nadir RGB satellite/drone image → per-pixel **height above local ground (AGL/nDSM),
in metres**, float32.

**AGL is not elevation.** `DSM = DTM + AGL`. Terrain is **added**, never subtracted.
See `SPEC.md` for the full output contract.

## Architecture

| part | what | inherited from |
|---|---|---|
| encoder | DINOv2 ViT-L/14, 24 blocks, hidden 1024, taps at layers 5/12/18/24 | Depth Anything V2 Large |
| neck | DPT, `reassemble_factors [4,2,1,0.5]`, fusion width 256 | Depth Anything V2 Large |
| head | **replaced** — adaptive bins (128), FiLM on log(GSD), auxiliary semantic (7 cls) | trained from scratch |

The DAv2 depth head is **discarded, not calibrated**. Its training loss is
affine-invariant by construction, so absolute scale was optimised away rather than lost,
and no rescaling recovers it. Measured on our own data (A0): stock DAv2 with a single
global affine reaches r 0.405; with a per-tile *oracle* affine it reaches 0.817. The
gap is the part rescaling cannot reach.

Losses: SILog (α 10, λ 0.85, on `log(h+1)` — AGL ground is exactly 0) + L1 + multi-scale
gradient (4 scales) + bin CE + chamfer + semantic CE. **Shadow-consistency loss is
implemented but OFF** (`w_shadow: 0.0`) — see the negative result below.

## Training

- Data: **GAMUS** (arXiv 2305.14914), HuggingFace `earthflow/GAMUS` — 8,724 tiles,
  1024², GSD 0.33 m, cities DC / NYC / PHL.
- Official split (NYC in train *and* test → in-domain), 40 epochs, batch 8, bf16.
- All three parameter groups trainable: encoder 1e-5, neck 5e-5, heads 1e-4;
  backbone frozen for the first 2 epochs only.
- Scale augmentation 0.5–1.97 → **trained GSD coverage 0.165–0.65 m/px**.
- Hardware: 1× NVIDIA L40 46 GB.

## Results

### In-domain (official split — NYC is in training)

| metric | value |
|---|---|
| building MAE | **1.549 m** |
| overall MAE | 1.234 m |
| overall r | 0.929 |
| sharpness ratio | 0.084 |

### Held-out city — NYC never trained on (**lead with this one**)

| metric | value |
|---|---|
| building MAE | **1.585 m** (best) / 1.597 final |
| building r | **0.901** |
| building bias | −0.387 m |
| overall MAE | 2.849 m |
| overall r | 0.739 |
| overall bias | **−2.019 m** |
| sharpness ratio | 0.023 |

Buildings transfer almost perfectly — 2.3% gap, correlation 0.901 vs 0.912. But a
**−2.02 m systematic offset** appears on the new city and overall correlation drops
0.93 → 0.74. The offset is what a scale+offset calibration removes downstream; the
correlation drop is structure and is not fixable that way.

**Caveat, state it every time:** measured on the 400-tile `holdout_val` set, which
`train.py` also checkpoint-selects on. Mildly optimistic. NYC is a *different urban
fabric*, not a taller one (max 99.7 m).

### Unseen-split spot checks

- Content-rich median over 428 unseen **test** tiles: **1.600 m**; val independently
  gives 1.560 m — two unseen splits agreeing to 2.5%.
- Best 12 of those 428: mean building MAE 0.739 m.
- **Worst tile in the test pool: 46.6 m.** Know this before you are asked.
- Train-pool median 1.339 vs test 1.600 → a **19% memorisation gap**. That is the
  honest overfitting number.

## What did NOT work — reported, not buried

- **Shadow-consistency loss buys 0.4%.** A5 1.841 → control (w_shadow=0) 1.739 →
  shadow 1.732. A6's apparent gain was 40 extra epochs at a decayed LR, which the
  control isolates. The loss is mechanically sound (occlusion IoU 0.91–0.93 against an
  independent numpy reimplementation, sign convention verified, gradients flow, term
  live at ~0.72 throughout). It simply does not move the metric on GAMUS at weight 0.1
  supervising 38% of tiles.
- **The A1–A5 ladder sits in a 1.769–1.848 band — 4.3%, single seed, no repeats.
  Those configurations cannot be ranked from that data.**
- **A7 (backbone scale) is the only unambiguous win**: 1.549 is 12% below the best
  ViT-S, several times the noise band, and achieved on 40 epochs against A5's 60 — the
  confound runs *against* it.

## Known limitations

- **Sharpness ratio 0.084–0.133 against 1.0 for truth.** Calibration by degrading the
  ground truth puts the output at roughly truth blurred at **σ ≈ 3 px ≈ 2 m**. Four
  approaches have failed to move it. Open problem; see `FINETUNE_INDIA_PLAN.md` and
  `HANDOFF_SESSION2.md` §4.
- **Trained GSD band is 0.165–0.65 m/px.** 1 m Cartosat-2S MX is outside it. Cartosat-3
  pan (0.25 m) is inside. `scripts/predict_external.py` resamples before inference —
  bypass that and the number is meaningless.
- **The outer ~130 px of an exported scene is ~8% worse** on building MAE (less overlap
  coverage, and a shadow falling outside the frame cannot be used). Mirror-padding in
  `infer.py` would fix it; not done.
- **No Indian validation whatsoever.** The external-image runs are qualitative only —
  no ground truth, assumed GSD, one image possibly AI-generated, one obliquely angled.
- **Honest accuracy claim: 2–4 m MAE.** Not a LiDAR replacement.

## Downstream contract

Consumers require, and this model provides: **float32** (8-bit quantisation creates tied
values and degenerate critical points in persistent homology), **unsmoothed**
(smoothing is the TDA stage's job), explicit nodata `-9999.0`, and globally consistent
tiling (seams read as high-persistence structure).

## Reproducing

```
python -m heightmap.train --config configs/ablations/a7_vitl.yaml
```

Checkpoint lives on the cluster at `outputs/a7_vitl/best.pt`. It is **not** in git —
`.gitignore` excludes `*.pt`, deliberately: a ViT-L checkpoint is far past what git
should carry, and a repo that stores weights becomes unclonable. Attach it to the
GitHub release for this tag instead, or keep it on the cluster and cite the path.

## Citation of the data

GAMUS: Xiong et al., *GAMUS: A Geometry-aware Multi-modal Semantic Segmentation
Benchmark for Remote Sensing Data*, arXiv:2305.14914.
Depth Anything V2: Yang et al., 2024.
