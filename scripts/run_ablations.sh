#!/usr/bin/env bash
# The full ablation ladder.  Run on the rented GPU, inside tmux.
#   bash scripts/run_ablations.sh data/GAMUS
set -euo pipefail
ROOT="${1:-data/GAMUS}"
PY="${PY:-python}"

echo "== A0: zero-shot baseline (the number that justifies everything) =="
$PY scripts/zero_shot_eval.py --root "$ROOT" --split val --out outputs/a0_zero_shot.json

for a in a1_regression a2_bins a3_gsd a4_semantic a5_domainrand; do
  echo "== ${a} =="
  $PY -m heightmap.train --config="configs/ablations/${a}.yaml" "data.root=${ROOT}"
done

echo "== A6: shadow consistency (needs sun angles; fine-tunes from A5) =="
if [ ! -f "${ROOT}/sun_angles.json" ]; then
  $PY scripts/estimate_sun.py --root "$ROOT" --out "${ROOT}/sun_angles.json"
fi
$PY -m heightmap.train --config=configs/ablations/a6_shadow.yaml \
    "data.root=${ROOT}" "data.sun_cache=${ROOT}/sun_angles.json"

echo "== A7: ViT-L =="
$PY -m heightmap.train --config=configs/ablations/a7_vitl.yaml "data.root=${ROOT}"

echo "== city-held-out (train DC+PHL, eval NYC) -- report this FIRST =="
$PY -m heightmap.train --config=configs/ablations/holdout.yaml "data.root=${ROOT}"

echo "== results =="
$PY scripts/collect_results.py
