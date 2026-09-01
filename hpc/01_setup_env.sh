#!/bin/bash
# RUN ON THE LOGIN NODE (needs internet).
#
# Standalone venv -- deliberately NOT --system-site-packages.  Inheriting the shared
# env looked cheap but pip then skips any dependency the shared env already satisfies,
# so torch 2.6 landed in the venv while its matching nvidia-cudnn stayed outside it and
# `import torch` died on libcudnn.so.9.  Downloading ~3.5 GB once removes that entire
# class of failure; the login node does ~25 MB/s so it costs a few minutes.
set -euo pipefail
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/depthwizard}"
CU_TAG="${CU_TAG:-cu124}"          # driver 555.42 = CUDA 12.5, so cu124 is supported

source /apps/anaconda3/bin/activate deeplearning
PY_BIN=$(which python)
source /apps/anaconda3/bin/activate base 2>/dev/null || true

rm -rf "$ENV_PREFIX"
"$PY_BIN" -m venv "$ENV_PREFIX"
source "$ENV_PREFIX/bin/activate"
python -m pip install --upgrade pip wheel

# torch first, on its own index, so it pulls its own matching nvidia-* stack
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CU_TAG}"
python -m pip install transformers safetensors huggingface_hub \
                      numpy scipy pillow h5py rasterio opencv-python-headless \
                      matplotlib omegaconf tqdm imageio

echo
echo "==================== VERIFY ===================="
python - <<'PY'
import sys
import torch
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda build", torch.version.cuda)
import torch.backends.cudnn as cudnn
print("cudnn       ", cudnn.version())
import transformers
print("transformers", transformers.__version__)
from transformers import AutoModelForDepthEstimation
print("AutoModelForDepthEstimation: importable")
for m in ("omegaconf","h5py","rasterio","cv2","numpy","scipy","matplotlib","imageio","tqdm"):
    mod=__import__(m); print(f"  {m:12s}", getattr(mod,"__version__","ok"))
print("\nENV OK  (CUDA availability is only testable on a compute node)")
PY
echo
echo "env at:   $ENV_PREFIX"
echo "activate: source $ENV_PREFIX/bin/activate"
