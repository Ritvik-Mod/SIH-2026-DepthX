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
# only what the shared env lacks; cv2/rasterio/h5py/numpy are inherited
python -m pip install omegaconf tqdm imageio

# transformers 5.x needs torch >= 2.5, and the shared env pins 2.4.1.  Installing the
# newer transformers alone makes it silently disable PyTorch entirely --
#   "[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.1"
# -- so AutoModelForDepthEstimation stops existing and every model call fails.
#
# Put a matching torch INTO the venv, where it shadows the shared 2.4.1.  cu124 rather
# than cu121: the driver here is 555.42 (CUDA 12.5), so cu124 is both supported and
# better tuned for the L40's Ada cores.  torchvision is installed alongside so the two
# stay version-matched.
CU_TAG="${CU_TAG:-cu124}"
python -m pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/${CU_TAG}"
python -m pip install --upgrade transformers safetensors huggingface_hub

echo
python - <<'PY'
import torch, transformers, sys
# the decisive check: transformers must actually have PyTorch enabled
try:
    from transformers import AutoModelForDepthEstimation
    print("AutoModelForDepthEstimation: importable  <-- torch/transformers are compatible")
except Exception as e:
    raise SystemExit(f"FATAL: {type(e).__name__}: {e}\n"
                     "transformers has disabled PyTorch -- the versions do not match.")
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
