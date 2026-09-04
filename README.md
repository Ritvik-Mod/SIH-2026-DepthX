# DepthWizard — single-image height estimation (SIH26175, ISRO)

Estimates per-pixel **height above local ground (AGL)** in metres from one nadir optical
image, and exports a georeferenced raster for the calibration → TDA → 3D stages.

**Interface contract: [`SPEC.md`](SPEC.md).** Read that first if you consume the output.

## What this is

DINOv2 encoder + DPT decoder, both inherited from Depth Anything V2, with the depth head
**replaced**. Stock DAv2 predicts affine-invariant inverse depth — its loss was built so
that `a·d + b` scores identically to `d`, so absolute scale was not lost, it was
optimised away. No post-hoc rescaling recovers it; the head has to be retrained against
metric nDSM supervision.

On top of that:

- **adaptive-bins head** — a pixel at a roof edge is either ground or roof, never the
  average; regression is forced to emit that average, which is what makes edges melt
- **GSD conditioning (FiLM)** — a 40-pixel-wide flat rectangle is a shed at 0.3 m/px and
  a warehouse at 2 m/px; this is what makes metric output possible from one image
- **auxiliary semantic head** — regularises the shared features and yields the building
  mask needed for building-only metrics
- **shadow-consistency loss** — physics, not a learned prior, so it transfers across
  continents and needs no height labels
- **uncertainty head** — optional per-pixel σ, for a tool whose wrong answers matter

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch torchvision transformers pillow numpy \
  scipy rasterio matplotlib opencv-python-headless huggingface_hub safetensors \
  h5py omegaconf tqdm imageio
```

## Quick start

```bash
# 0. sample output for the downstream stages -- no model needed
python scripts/make_contract_sample.py

# 1. synthetic tiles, so everything is testable before the real download
python scripts/make_synthetic.py --per-split 6

# 2. the seven checks.  Do not rent a GPU until all pass.
python scripts/sanity.py

# 3. real data
python scripts/download_gamus.py --out data/GAMUS
python scripts/dataset_stats.py --root data/GAMUS      # sets h_max, confirms SILog shift
python scripts/estimate_sun.py --root data/GAMUS       # needed only for A6

# 4. the baseline that justifies the project
python scripts/zero_shot_eval.py --root data/GAMUS --split val

# 5. train
python -m heightmap.train --config=configs/ablations/a5_domainrand.yaml
python -m heightmap.train --config=configs/ablations/holdout.yaml   # the honest number

# 6. predict
python -m heightmap.predict --ckpt outputs/a5_domainrand/best.pt --image scene.tif \
  --out outputs/predictions --tta
```

## Ablation ladder

Each config states only its differences from the one below it (`_base_:` inheritance).

| # | Config | Adds | Tests |
|---|---|---|---|
| A0 | `scripts/zero_shot_eval.py` | — | ceiling for any rescaling of stock DAv2 |
| A1 | `a1_regression.yaml` | fine-tuning, regression head | does replacing the head help |
| A2 | `a2_bins.yaml` | adaptive bins | long-tail flattening (watch signed bias) |
| A3 | `a3_gsd.yaml` | GSD conditioning + scale aug | metric scale awareness |
| A4 | `a4_semantic.yaml` | semantic aux head | multi-task benefit |
| A5 | `a5_domainrand.yaml` | domain randomisation | robustness |
| A6 | `a6_shadow.yaml` | shadow consistency | physical grounding |
| A7 | `a7_vitl.yaml` | ViT-L | capacity |
| — | `holdout.yaml` | train DC+PHL, eval NYC | **generalisation** |

Run every row on **both** splits. The gap between them is itself a finding, and it is
the number that predicts performance on unseen landscapes.

## Metrics

Reported overall **and** building-only, because ~a third of pixels are ground at exactly
0 m and are nearly free.

- `MAE`, `RMSE` — structure *and* calibration
- `r` (Pearson) — **invariant to scale and shift**, so it isolates structure. Since
  metric anchoring is a separate stage downstream, this is the cleanest measure of this
  model's own contribution
- `bias` — **signed**. Two models can both score 4 m MAE: one randomly off by ±4 m, the
  other consistently 4 m short. Only bias separates them, and the second is the
  long-tail flattening problem, quantified
- `δ₁` — computed on `(h+1)` so ground does not divide by zero. Stated in the output
- `sharpness` — mean gradient at *true-edge* locations. Note that mean gradient alone
  does **not** measure sharpness: total variation is preserved when a step is blurred
  into a ramp, so the measurement must be conditioned on where the truth has edges

## Layout

```
configs/            base.yaml + ablations/ (A1..A7, holdout)
heightmap/
  config.py         yaml + dotted overrides + _base_ inheritance + validation guards
  data/             gamus.py (dataset, splits), transforms.py (aug, scale->GSD)
  models/           net.py (HeightNet), heads.py (bins, FiLM, semantic, uncertainty)
  losses/           terms.py (silog, l1, gradient, bins, chamfer, nll), shadow.py, combined.py
  metrics.py        metrics + exact streaming accumulator
  train.py  infer.py  predict.py  export.py
scripts/            download_gamus, dataset_stats, estimate_sun, zero_shot_eval,
                    make_synthetic, make_contract_sample, sanity
tests/              test_data, test_model, test_shadow, test_metrics, test_infer
```

## Things the code knows that are easy to get wrong

- **h5py handles do not survive fork or pickling.** `GamusDataset.__getstate__` drops the
  cache so DataLoader workers reopen their own. Without it you get corrupted reads or a
  hang, not a clean error.
- **Flips change the apparent sun direction.** `config.py` refuses `w_shadow > 0` together
  with `aug.geometric`, because the loss would punish correct predictions.
- **Constant GSD leaves FiLM's first layer untrained.** At `gsd == gsd_ref`, `z = log(1) = 0`
  and `dL/dW = dL/dout · z = 0`. Scale augmentation is not optional; `sanity.py` asserts
  this property explicitly.
- **MPS cannot do adaptive pooling for non-divisible sizes** (518 → 296), so the target
  downsamples use bilinear.
- **bf16, not fp16.** SILog takes logs and square roots; fp16's narrow range NaNs them.
- **GAMUS classes are 1..6, and building is 3** (1 ground, 2 low-vegetation, 3 building,
  4 water, 5 road, 6 tree; 0 is unlabelled). Using 2 for building silently measures
  low-vegetation, and excluding the wrong ids from the shadow loss removes the very
  occluders that carry the height signal. See `heightmap/data/classes.py`.
- **GAMUS GSD is 0.33 m** (paper, sec. 1), so trained GSD coverage is 0.17–0.65 m/px.
  Coarser inputs (1 m Cartosat MX) are outside that range.
- **Uniform bin probabilities start the prediction at `h_max/2`** (~125 m for a scene whose
  median height is 4 m). The bin logits are initialised to an exponential decay whose
  expectation is `init_height`, derived rather than tuned.
- **Tile drift comes from tiles with no visible ground.** The model must locate the ground
  to predict height above it; a tile that is entirely rooftop has to guess. Hence
  coarse-guided levelling in `infer.py`. 
  Deployment fix 1.
  Deployment fix 2.
  Deployment fix 3.
