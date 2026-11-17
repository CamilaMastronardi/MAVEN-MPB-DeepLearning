#!/usr/bin/env bash
# Recalibrate ONLY completed HP1 checkpoints on their inner-month monitors.
# Does NOT train models or query BigQuery.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_refiner_v10_hp2a}"
HP1_DIR="${HP1_DIR:-results/metrics/mag_swea_refiner_v10_hp1}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
FOLDS="${FOLDS:-completed}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
ALPHA_STEP="${ALPHA_STEP:-0.1}"

python -m py_compile \
  scripts/5_training/5f_train_mag_swea_refiner_v10.py \
  scripts/5_training/5m_recalibrate_v10_hp2a.py \
  tests/test_v10_hp2_calibration.py
python -m unittest discover -s tests -p 'test_v10_hp2_calibration.py' -v

python -u scripts/5_training/5m_recalibrate_v10_hp2a.py \
    --hp1-dir "$HP1_DIR" \
    --pretrained-dir "$PRETRAINED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --folds "$FOLDS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --torch-threads "$TORCH_THREADS" \
    --alpha-step "$ALPHA_STEP"
