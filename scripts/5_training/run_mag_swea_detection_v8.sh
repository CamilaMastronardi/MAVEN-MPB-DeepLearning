#!/usr/bin/env bash
# V8: warm-start V7b; denser TRAIN negatives; thresholded scan benchmark.
# Full 559-event CV by default; source caches are reused if complete.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_detection_v8}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-35}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"

echo "V8: detection-oriented fine-tuning of complete V7b checkpoints."
echo "70-min event-centered blocks; thresholded independent peaks; not full-orbit detection."
echo "No new source downloads when original 559 MAG/SWEA caches are complete."
echo "Outputs: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
    echo "PILOT SUBSET: $PILOT_EVENTS_PER_FOLD events per outer fold."
    echo "WARNING: V7b initialization is from FULL training, including these"
    echo "pilot validation months in some folds? Corresponding fold models"
    echo "always exclude THEIR OWN test fold, but metrics are only subset CV."
fi

python -m py_compile \
  src/detection/mpb_scan_detection.py \
  scripts/5_training/5c_train_mag_swea_detection_v8.py
python -m unittest discover -s tests -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5c_train_mag_swea_detection_v8.py \
    --folds all \
    --pretrained-dir "$PRETRAINED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
    --max-epochs "$MAX_EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --max-query-gib "$MAX_QUERY_GIB" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
