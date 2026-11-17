#!/usr/bin/env bash
# V8 conditional localizer: fine-tune the completed V7b 4-fold models.
# No known-MPB-vs-no-MPB classification is attempted here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_localization_v8}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MAX_EPOCHS="${MAX_EPOCHS:-28}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V8: LOCALIZATION given known MPB, NOT 70-min presence detection."
echo "Warm-start V7b full-fold models, original cached MAG/SWEA measurements."
echo "New nearby ordinal ranking; ambiguous neighboring windows are unlabeled."
echo "Four outer folds; comparison on the exact same test events as V7b."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
    echo "WARNING: pilot subset only, not complete outer cross-validation."
    echo "Pretrained initialization remains the FULL V7b fold checkpoint."
fi

python -m py_compile \
  src/detection/mpb_localization_refinement.py \
  scripts/5_training/5d_train_mag_swea_localization_v8.py
python -m unittest discover -s tests -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5d_train_mag_swea_localization_v8.py \
    --folds all \
    --pretrained-dir "$PRETRAINED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    --max-epochs "$MAX_EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
    --max-query-gib "$MAX_QUERY_GIB" \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
