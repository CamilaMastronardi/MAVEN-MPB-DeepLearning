#!/usr/bin/env bash
# V14: V10 coarse-to-fine with TRAINABLE 8-conv 3D MAG and SWEA bottlenecks.
# Frozen matching-fold V7b coarse proposal, plus reconstruction supervision.
# Run offline cache preflight first. Never overwrite V10/V13 checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
if [[ "$PILOT_EVENTS_PER_FOLD" == "0" ]]; then
    DEFAULT_OUTPUT="results/metrics/v14_dual_3d_refiner"
else
    DEFAULT_OUTPUT="results/metrics/v14_dual_3d_refiner_pilot"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MANIFEST="${MANIFEST:-data/splits/cv4_monthly_manifest.csv}"
FOLDS="${FOLDS:-all}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
MIN_EPOCHS="${MIN_EPOCHS:-20}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LOCALIZATION_WEIGHT="${LOCALIZATION_WEIGHT:-3.0}"
OFFSET_WEIGHT="${OFFSET_WEIGHT:-1.5}"
RECONSTRUCTION_WEIGHT="${RECONSTRUCTION_WEIGHT:-0.2}"
GAUSSIAN_SIGMA_S="${GAUSSIAN_SIGMA_S:-24}"
TRAIN_POSITIVES_PER_EVENT="${TRAIN_POSITIVES_PER_EVENT:-6}"
TRAIN_NEGATIVES_PER_EVENT="${TRAIN_NEGATIVES_PER_EVENT:-3}"
ALPHA_STEP="${ALPHA_STEP:-0.1}"
CALIBRATION_GOAL="${CALIBRATION_GOAL:-within_60}"

echo "V14: frozen V7b candidate scan + two trainable 8-layer 3D CNN AEs."
echo "Loss: BCE + KL*$LOCALIZATION_WEIGHT + offset*$OFFSET_WEIGHT + reconstruction*$RECONSTRUCTION_WEIGHT."
echo "Folds: $FOLDS, pilot events/fold: $PILOT_EVENTS_PER_FOLD"
echo "Output: $OUTPUT_DIR. Real cache data only; no BigQuery downloads."

python -m py_compile \
  src/detection/mpb_v14_dual_bottleneck_refiner.py \
  scripts/5_training/5n_train_v14_dual_bottleneck.py \
  tests/test_v14_dual_bottleneck.py
python -m unittest discover -s tests -p 'test_v14_dual_bottleneck.py' -v

# This must succeed before the trainer can load any scientific source data.
python scripts/5_training/5l_preflight_v10_hp1_cache.py \
  --manifest "$MANIFEST" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD"

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5n_train_v14_dual_bottleneck.py \
  --manifest "$MANIFEST" \
  --pretrained-dir "$PRETRAINED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --folds "$FOLDS" \
  --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
  --max-epochs "$MAX_EPOCHS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --torch-threads "$TORCH_THREADS" \
  --learning-rate "$LEARNING_RATE" \
  --localization-weight "$LOCALIZATION_WEIGHT" \
  --offset-weight "$OFFSET_WEIGHT" \
  --reconstruction-weight "$RECONSTRUCTION_WEIGHT" \
  --gaussian-sigma-s "$GAUSSIAN_SIGMA_S" \
  --train-positives-per-event "$TRAIN_POSITIVES_PER_EVENT" \
  --train-negatives-per-event "$TRAIN_NEGATIVES_PER_EVENT" \
  --alpha-step "$ALPHA_STEP" \
  --calibration-goal "$CALIBRATION_GOAL" \
  --resume --execute \
  2>&1 | tee -a "$OUTPUT_DIR/training.log"
