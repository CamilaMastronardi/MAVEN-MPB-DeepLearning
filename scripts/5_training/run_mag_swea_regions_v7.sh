#!/usr/bin/env bash
# Seventh experiment: MAG+SWEA plasma-region and MPB-transition learning.
# All checkpoints/metrics are isolated from 1-6, including the sixth run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_regions_v1}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVALUATION_STRIDE_S="${EVALUATION_STRIDE_S:-32}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
REFRESH_SWEA_CACHE="${REFRESH_SWEA_CACHE:-0}"

echo "Running offline MAG+SWEA dataset, model, cache and regression tests..."
python -m py_compile \
    src/data/swea_event_cache.py \
    src/data/mpb_weak_regions.py \
    src/data/mpb_dual_windows.py \
    src/detection/mag_swea_region_model.py \
    scripts/5_training/5b_train_mag_swea_regions.py
python -m unittest discover -s tests -v

if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
    echo "WARNING: PILOT SUBSET ($PILOT_EVENTS_PER_FOLD events/fold), not full CV."
fi
if [[ "$REFRESH_SWEA_CACHE" != "0" && "$REFRESH_SWEA_CACHE" != "1" ]]; then
    echo "REFRESH_SWEA_CACHE must be 0 or 1" >&2
    exit 2
fi

extra=()
if [[ "$REFRESH_SWEA_CACHE" == "1" ]]; then
    extra+=(--refresh-swea-cache)
fi

echo "V7: 4-fold MAG+SWEA weak plasma regions and MPB transition classification."
echo "MAG 1s, SWEA native ~32s, physical energy 30-300eV; no time interpolation."
echo "AE auxiliary weight 0.15, region BCE weight 0.5, transition BCE weight 1."
echo "Early-stop min $MIN_EPOCHS patience $PATIENCE max $MAX_EPOCHS."
echo "Outer MPB detection threshold fitted on inner-month monitor only."
echo "Cumulative BigQuery dry-run cap: $MAX_TOTAL_GIB GiB."
echo "Source caches MAG=$MAG_CACHE_DIR SWEA=$SWEA_CACHE_DIR."
echo "Distinct run output=$OUTPUT_DIR."

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5b_train_mag_swea_regions.py \
    --folds all \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
    --max-epochs "$MAX_EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --evaluation-stride-s "$EVALUATION_STRIDE_S" \
    --max-query-gib "$MAX_QUERY_GIB" \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    "${extra[@]}" \
    --output-dir "$OUTPUT_DIR" --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
