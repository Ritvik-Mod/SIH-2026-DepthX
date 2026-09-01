#!/bin/bash
# RUN ON THE LOGIN NODE.  Pre-stages everything that needs internet, so compute nodes can
# run fully offline (they usually cannot reach the outside world).
#   bash hpc/02_fetch_assets.sh /path/to/data_root
set -euo pipefail
DATA_ROOT="${1:?usage: 02_fetch_assets.sh <data_root>}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/depthwizard}"
source /apps/anaconda3/bin/activate "$ENV_PREFIX"

# Keep the HF cache inside the project so it is on the same (large) filesystem as the data
export HF_HOME="$DATA_ROOT/hf_cache"
mkdir -p "$HF_HOME" "$DATA_ROOT/GAMUS"

echo "== 1/2  backbone checkpoints (small, ~100 MB each) =="
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in ["depth-anything/Depth-Anything-V2-Small-hf",
             "depth-anything/Depth-Anything-V2-Base-hf",
             "depth-anything/Depth-Anything-V2-Large-hf"]:
    p = snapshot_download(repo)
    print("  cached:", repo, "->", p)
PY

echo
echo "== 2/2  GAMUS dataset (~80 GB, repacked to ~50 GB afterwards) =="
echo "   this takes a while; run it inside tmux/screen so a dropped SSH does not kill it"
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("earthflow/GAMUS", repo_type="dataset",
                      local_dir="$DATA_ROOT/GAMUS", max_workers=8)
print("  dataset at:", p)
PY

echo
echo "staged. now run:  qsub hpc/10_repack.pbs   (frees ~30 GB)"
echo "and export these in every job script:"
echo "   export HF_HOME=$HF_HOME"
echo "   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
