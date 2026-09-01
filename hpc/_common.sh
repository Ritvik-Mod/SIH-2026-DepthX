# Sourced by every PBS job.  Single place to change paths/env.
set -euo pipefail
: "${PROJECT:=$HOME/SIH2026}"
: "${DATA_ROOT:=$PROJECT/data}"
: "${ENV_PREFIX:=$HOME/envs/depthwizard}"

cd "${PBS_O_WORKDIR:-$PROJECT}"

# PBS Pro spools stdout on the execution node and only copies it to the -o path when
# the job ENDS, so `tail -f` on that file shows nothing for the whole run.  Mirror
# everything to a live log on the shared filesystem instead, which IS tailable.
mkdir -p logs
LIVE_LOG="logs/${PBS_JOBNAME:-job}.live"
exec > >(tee -a "$LIVE_LOG") 2>&1
echo "=== live log: $LIVE_LOG  (tail -f this, not the -o file) ==="
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

# Fail fast on an incomplete upload rather than 40 lines into a traceback.
for _m in heightmap/data heightmap/models heightmap/losses heightmap/utils; do
  [ -d "$_m" ] || { echo "FATAL: $_m missing -- the rsync dropped it."; \
                    echo "  a bare --exclude 'data' also matches heightmap/data; use --exclude '/data'"; \
                    exit 1; }
done

echo "host=$(hostname)  job=${PBS_JOBID:-none}  env=$ENV_PREFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
