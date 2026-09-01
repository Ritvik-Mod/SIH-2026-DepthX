#!/bin/bash
# RUN ON THE LOGIN NODE (needs internet).
#
# The shared 'deeplearning' env already has a working CUDA stack:
#     python 3.11.13, torch 2.4.1+cu121, cuda avail True, bf16 True,
#     transformers / h5py / rasterio / cv2 present, only omegaconf missing.
# So we layer a venv ON TOP of it with --system-site-packages rather than building a
# fresh conda env.  That reuses the ~3 GB torch install, takes a minute instead of an
# hour, and still lets us pin/upgrade the few packages we actually care about without
# writing to an env other students share.
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/depthwizard}"

source /apps/anaconda3/bin/activate deeplearning
python -m venv --system-site-packages "$ENV_PREFIX"
source "$ENV_PREFIX/bin/activate"

python -m pip install --upgrade pip
# only what the shared env lacks; torch/cv2/rasterio/h5py are inherited
python -m pip install omegaconf tqdm imageio
# transformers must be new enough for DepthAnythingForDepthEstimation's
# backbone/neck/head split -- 04_probe.pbs is what actually proves it
python -m pip install --upgrade "transformers>=4.45" safetensors huggingface_hub

echo
python - <<'PY'
import torch, transformers, sys
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda build", torch.version.cuda)
print("transformers", transformers.__version__)
for m in ("omegaconf","h5py","rasterio","cv2","tqdm","imageio","numpy","scipy"):
    try:
        mod=__import__(m); print(f"{m:12s}", getattr(mod,"__version__","ok"))
    except Exception as e:
        print(f"{m:12s} MISSING {e}")
PY
echo
echo "env at: $ENV_PREFIX"
echo "activate: source $ENV_PREFIX/bin/activate"
