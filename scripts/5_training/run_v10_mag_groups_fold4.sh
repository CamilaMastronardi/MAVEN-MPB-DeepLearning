#!/usr/bin/env bash
# Train TWO group-specific V10 refiners, testing only FOLD 4 (30+109 events).
# Reuse frozen V7b fold-4 teacher and original MAG/SWEA source caches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/v10_mag_group_specialists_fold4}"
MAG_GROUPS_CSV="${MAG_GROUPS_CSV:-results/metrics/mag_autoencoder_clusters_v13/crossing_latents_groups.csv}"
PRETRAINED_DIR="${PRETRAINED_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
BASELINE_V10_DIR="${BASELINE_V10_DIR:-results/metrics/mag_swea_refiner_v10}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MAX_EPOCHS="${MAX_EPOCHS:-45}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-9}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V10 per MAG group: ONLY outer FOLD 4 evaluation."
echo "TRAIN: folds 1+2+3, separately for MAG1 and MAG2."
echo "V7b fold-4 source teacher FROZEN, refinement heads trained separately."
echo "Inner month monitor selected separately per group (>=5 events)."
echo "Routing uses catalog-timed V13 MAG group IDs; exploratory oracle grouping."
echo "Output: $OUTPUT_DIR"

test -f "$MAG_GROUPS_CSV" || {
  echo "Missing MAG cluster label CSV: $MAG_GROUPS_CSV" >&2
  exit 1
}

python -m py_compile \
  scripts/5_training/5k_train_v10_mag_groups_fold4.py \
  tests/test_v10_mag_groups_fold4.py

python -m unittest discover -s tests -p 'test_v10_mag_groups_fold4.py' -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5k_train_v10_mag_groups_fold4.py \
  --mag-groups-csv "$MAG_GROUPS_CSV" \
  --pretrained-dir "$PRETRAINED_DIR" \
  --baseline-v10-dir "$BASELINE_V10_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --torch-threads "$TORCH_THREADS" \
  --max-query-gib "$MAX_QUERY_GIB" \
  --max-total-gib "$MAX_TOTAL_GIB" \
  --groups all --resume --execute \
  2>&1 | tee -a "$OUTPUT_DIR/training.log"
