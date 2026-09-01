# Sourced by every PBS job.  Single place to change paths/env.
set -euo pipefail
: "${PROJECT:=$HOME/SIH2026}"
: "${DATA_ROOT:=$PROJECT/data}"
: "${ENV_PREFIX:=$HOME/envs/depthwizard}"

cd "${PBS_O_WORKDIR:-$PROJECT}"
source /apps/anaconda3/bin/activate "$ENV_PREFIX"

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
