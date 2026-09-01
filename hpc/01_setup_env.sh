#!/bin/bash
# RUN ON THE LOGIN NODE (it needs internet).  Builds a self-contained conda env so we do
# not depend on whatever version the shared 'deeplearning' env happens to have.
#
#   bash hpc/01_setup_env.sh <CUDA_TAG>       e.g. cu121, cu118, cu124
# Pick the tag from 00_discover output: it must be <= the driver's CUDA version.
set -euo pipefail
CUDA_TAG="${1:?usage: 01_setup_env.sh <cu118|cu121|cu124>}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/depthwizard}"

source /apps/anaconda3/bin/activate base
conda create -y -p "$ENV_PREFIX" python=3.11
source /apps/anaconda3/bin/activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
python -m pip install transformers h5py numpy scipy pillow opencv-python-headless \
                     rasterio matplotlib omegaconf tqdm imageio huggingface_hub safetensors

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda build", torch.version.cuda)
print("env OK -- CUDA availability is only testable on a compute node")
PY
echo
echo "env at: $ENV_PREFIX"
echo "activate with: source /apps/anaconda3/bin/activate $ENV_PREFIX"
