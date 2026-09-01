#!/bin/bash
# Queue the whole ablation ladder as a dependency CHAIN (one GPU at a time, in order).
# A chain rather than parallel jobs because a student allocation usually has a low
# max_run limit, and because each stage tells you whether the next is worth running.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PREV=""
for CFG in configs/ablations/a1_regression.yaml \
           configs/ablations/a2_bins.yaml \
           configs/ablations/a3_gsd.yaml \
           configs/ablations/a4_semantic.yaml \
           configs/ablations/a5_domainrand.yaml \
           configs/ablations/a6_shadow.yaml \
           configs/ablations/a7_vitl.yaml \
           configs/ablations/holdout.yaml ; do
  if [ -z "$PREV" ]; then
    JID=$(qsub -v CFG="$CFG" -N "dw_$(basename "$CFG" .yaml)" hpc/30_train.pbs)
  else
    JID=$(qsub -W depend=afterany:"$PREV" -v CFG="$CFG" -N "dw_$(basename "$CFG" .yaml)" hpc/30_train.pbs)
  fi
  echo "queued $(basename "$CFG" .yaml)  ->  $JID"
  PREV="$JID"
done
echo; echo "watch with:  qstat -u \$USER -a"
