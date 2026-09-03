#!/bin/bash
# Chain the F0-F3 block sweep, two seeds each, one GPU at a time.
#
# Eight runs.  A single seed CANNOT rank these: the A1-A5 ladder sat in a 1.769-1.848
# band -- 4.3% -- on one seed, and that is why those five configurations still cannot
# be ordered.  Do not cut the second seed to save time; a sweep that cannot be read is
# slower than one that takes twice as long.
set -euo pipefail
cd "$(dirname "$0")/.."
PREV=""
for SEED in 1337 2024; do
  for ARM in f0_headsonly f1_early f2_late f3_full; do
    V="CFG=configs/finetune/${ARM}.yaml,EXTRA=seed=${SEED} out_dir=outputs/${ARM}_s${SEED}"
    if [ -z "$PREV" ]; then
      JID=$(qsub -N "dw_${ARM:0:6}$SEED" -v "$V" hpc/30_train.pbs)
    else
      JID=$(qsub -W "depend=afterany:$PREV" -N "dw_${ARM:0:6}$SEED" -v "$V" hpc/30_train.pbs)
    fi
    echo "queued $ARM seed $SEED -> $JID"
    PREV="$JID"
  done
done
echo
echo "8 jobs chained.  Watch with: qstat -u \$USER -a"
echo "Collect with: python scripts/collect_results.py outputs/f*_s*"
