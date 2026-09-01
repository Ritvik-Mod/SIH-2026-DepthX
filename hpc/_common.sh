# Sourced by every PBS job.  Single place to change paths/env.
set -euo pipefail
: "${PROJECT:=$HOME/SIH2026}"
: "${DATA_ROOT:=$PROJECT/data}"
: "${ENV_PREFIX:=$HOME/envs/depthwizard}"

cd "${PBS_O_WORKDIR:-$PROJECT}"
source /apps/anaconda3/bin/activate deeplearning
source "$ENV_PREFIX/bin/activate"

# PBS here does not fence GPUs, so choose the least-used one ourselves unless the
# scheduler already pinned us. Without this we can land on a GPU another job is using.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used \
      --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
echo "using GPU ${CUDA_VISIBLE_DEVICES} of $(nvidia-smi -L | wc -l)"

# Compute nodes usually have no internet: force offline so a lookup fails loudly and
# fast instead of hanging until the walltime kills the job.
export HF_HOME="$DATA_ROOT/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)  job=${PBS_JOBID:-none}  env=$ENV_PREFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
