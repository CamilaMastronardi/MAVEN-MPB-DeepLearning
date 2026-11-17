#!/usr/bin/env bash
# V10: freeze full V7b plasma-region classifier and train local time refiner.
# Full four-fold MPB conditional localization by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_refiner_v10}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MAX_EPOCHS="${MAX_EPOCHS:-45}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-9}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V10: frozen V7b MAG+SWEA physical-region teacher and new offset head."
echo "Teacher weights, monitor, normalizers and folds stay isolated."
echo "Conditional localization: every 70-min test block has catalog MPB."
echo "New head trained on real shifted 10-min MAG/SWEA windows."
echo "Calibrate correction strength strictly on inner-month monitor."
echo "Source cached MAG/SWEA; do not overwrite V7b/V8/V9 results."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
    echo "WARNING: PILOT subset, not full four-fold CV."
fi

python -m py_compile \
  src/detection/mpb_region_anchored_refiner.py \
  scripts/5_training/5f_train_mag_swea_refiner_v10.py \
  tests/test_mag_swea_refiner_v10.py
python -m unittest discover -s tests -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5f_train_mag_swea_refiner_v10.py \
    --folds all \
    --pretrained-dir "$PRETRAINED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    --max-epochs "$MAX_EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --torch-threads "$TORCH_THREADS" \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
    --max-query-gib "$MAX_QUERY_GIB" \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
