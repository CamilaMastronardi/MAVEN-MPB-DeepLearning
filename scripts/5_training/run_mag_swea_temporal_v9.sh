#!/usr/bin/env bash
# V9 conditional MPB localization, pretrained V7b encoders + dense time head.
# Reuses MAG/SWEA cached 70-minute event blocks; no full-orbit negatives.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_temporal_v9}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MAX_EPOCHS="${MAX_EPOCHS:-40}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
TORCH_THREADS="${TORCH_THREADS:-2}"

echo "V9: dense temporal localization conditional on known MPB."
echo "MAG native 1-sec logits; SWEA native frame latent time alignment."
echo "Transfer fold-matched V7b encoders; center varied within 10-min clips."
echo "Source data: cached MAG+SWEA, no requery when cache is complete."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
    echo "PILOT subset: $PILOT_EVENTS_PER_FOLD events per fold, not full CV."
fi

python -m py_compile \
  src/detection/mpb_temporal_localizer.py \
  scripts/5_training/5e_train_mag_swea_temporal_v9.py \
  tests/test_mag_swea_temporal_v9.py
python -m unittest discover -s tests -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5e_train_mag_swea_temporal_v9.py \
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
    --torch-threads "$TORCH_THREADS" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
