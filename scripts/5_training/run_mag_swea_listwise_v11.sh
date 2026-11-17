#!/usr/bin/env bash
# V11: global ranking of every measured MAG+SWEA MPB candidate + V10 fine head.
# Known-MPB temporal localization ONLY. Reuses frozen V7b/V10 checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_listwise_v11}"
V7B_DIR="${V7B_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
V10_DIR="${V10_DIR:-results/metrics/mag_swea_refiner_v10}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MAX_EPOCHS="${MAX_EPOCHS:-70}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVENTS_PER_UPDATE="${EVENTS_PER_UPDATE:-12}"
TORCH_THREADS="${TORCH_THREADS:-2}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V11 FULL GRID: frozen V7b scores, learned listwise reranker, frozen V10 refinement."
echo "TARGET: >=80% within +/-60s; prior V10 53.0%. This is a testable goal, NOT a guarantee."
echo "Complete 70-min scans of 10-min MAG/SWEA windows at native cadence."
echo "Source measurements reused from event caches; V11 feature caching per fold."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
  echo "PILOT only: $PILOT_EVENTS_PER_FOLD event(s) per fold; not full CV."
fi

python -m py_compile \
  src/detection/mpb_listwise_reranker_v11.py \
  scripts/5_training/5g_train_mag_swea_listwise_v11.py \
  tests/test_mag_swea_listwise_v11.py

python -m unittest discover -s tests -p 'test_mag_swea_listwise_v11.py' -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5g_train_mag_swea_listwise_v11.py \
  --folds all \
  --v7b-dir "$V7B_DIR" \
  --v10-dir "$V10_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --events-per-update "$EVENTS_PER_UPDATE" \
  --torch-threads "$TORCH_THREADS" \
  --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
  --max-query-gib "$MAX_QUERY_GIB" \
  --max-total-gib "$MAX_TOTAL_GIB" \
  --resume --execute \
  2>&1 | tee -a "$OUTPUT_DIR/training.log"
