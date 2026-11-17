#!/usr/bin/env bash
# V10-HP1: hyperparameter trial of the ORIGINAL four-fold MAG+SWEA refiner.
# Keep V7b frozen; preserve CV splits, 10-min native windows and inner monitor.
# Distinct output directory. Read-only cache preflight; no BigQuery required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
if [[ "$PILOT_EVENTS_PER_FOLD" == "0" ]]; then
    DEFAULT_OUTPUT="results/metrics/mag_swea_refiner_v10_hp1"
else
    DEFAULT_OUTPUT="results/metrics/mag_swea_refiner_v10_hp1_pilot"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MANIFEST="${MANIFEST:-data/splits/cv4_monthly_manifest.csv}"

MAX_EPOCHS="${MAX_EPOCHS:-150}"
MIN_EPOCHS="${MIN_EPOCHS:-20}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
SEED="${SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LOCALIZATION_WEIGHT="${LOCALIZATION_WEIGHT:-2.0}"
OFFSET_WEIGHT="${OFFSET_WEIGHT:-1.0}"
GAUSSIAN_SIGMA_S="${GAUSSIAN_SIGMA_S:-32}"
TRAIN_POSITIVES_PER_EVENT="${TRAIN_POSITIVES_PER_EVENT:-4}"
TRAIN_NEGATIVES_PER_EVENT="${TRAIN_NEGATIVES_PER_EVENT:-3}"
EVALUATION_STRIDE_S="${EVALUATION_STRIDE_S:-32}"
FOLDS="${FOLDS:-all}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

# Guard against accidentally overwriting either reference experiment.
case "$OUTPUT_DIR" in
    results/metrics/mag_swea_refiner_v10|results/metrics/mag_swea_regions_ranking_v1)
        echo "ERROR: OUTPUT_DIR must be independent of the original V10/V7b." >&2
        exit 2 ;;
esac

echo "V10-HP1: V7b teacher frozen, no model architecture changes."
echo "max/min epochs=$MAX_EPOCHS/$MIN_EPOCHS patience=$PATIENCE"
echo "lr=$LEARNING_RATE KL=$LOCALIZATION_WEIGHT offset=$OFFSET_WEIGHT sigma=$GAUSSIAN_SIGMA_S"
echo "positive/negative windows per event=$TRAIN_POSITIVES_PER_EVENT/$TRAIN_NEGATIVES_PER_EVENT"
echo "outer folds=$FOLDS; pilot events per fold=$PILOT_EVENTS_PER_FOLD"
echo "output=$OUTPUT_DIR"
echo "Cache MUST be complete; abort before BigQuery if it is not."

python -m py_compile \
    src/detection/mpb_region_anchored_refiner.py \
    scripts/5_training/5f_train_mag_swea_refiner_v10.py \
    scripts/5_training/5l_preflight_v10_hp1_cache.py \
    tests/test_mag_swea_refiner_v10.py
python -m unittest discover -s tests -p 'test_mag_swea_refiner_v10.py' -v

python scripts/5_training/5l_preflight_v10_hp1_cache.py \
    --manifest "$MANIFEST" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD"

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5f_train_mag_swea_refiner_v10.py \
    --manifest "$MANIFEST" \
    --folds "$FOLDS" \
    --pretrained-dir "$PRETRAINED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    --max-epochs "$MAX_EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --torch-threads "$TORCH_THREADS" \
    --seed "$SEED" \
    --learning-rate "$LEARNING_RATE" \
    --localization-weight "$LOCALIZATION_WEIGHT" \
    --offset-weight "$OFFSET_WEIGHT" \
    --gaussian-sigma-s "$GAUSSIAN_SIGMA_S" \
    --train-positives-per-event "$TRAIN_POSITIVES_PER_EVENT" \
    --train-negatives-per-event "$TRAIN_NEGATIVES_PER_EVENT" \
    --evaluation-stride-s "$EVALUATION_STRIDE_S" \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
    --max-query-gib "$MAX_QUERY_GIB" \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
