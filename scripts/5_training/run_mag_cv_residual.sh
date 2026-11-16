#!/usr/bin/env bash
# Fifth FULL MAG experiment (v5): deeper residual temporal encoder +
# the v4 asymmetric transition detector. All 569 labeled events, all folds.
# An independent output directory; does not modify v4 running checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-20}"
MIN_EPOCHS="${MIN_EPOCHS:-25}"
TRAIN_WINDOWS_PER_EVENT="${TRAIN_WINDOWS_PER_EVENT:-600}"
TRAIN_WINDOWS_PER_EPOCH="${TRAIN_WINDOWS_PER_EPOCH:-24}"
VAL_CENTERS_PER_EVENT="${VAL_CENTERS_PER_EVENT:-600}"
SIGMA_SECONDS="${SIGMA_SECONDS:-60}"
LAMBDA_DETECTION="${LAMBDA_DETECTION:-10}"
LAMBDA_MAGNITUDE="${LAMBDA_MAGNITUDE:-1}"
TRANSITION_SIDE_OFFSET_S="${TRANSITION_SIDE_OFFSET_S:-120}"
TRANSITION_CENTER_EXCLUSION_S="${TRANSITION_CENTER_EXCLUSION_S:-15}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
REFRESH_MAG_CACHE="${REFRESH_MAG_CACHE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_cv_residual_transition_v1}"

echo "Checking v5 code, cache integrity tests and offline model tests..."
python -m py_compile \
    src/detection/mag_autoencoder.py \
    src/detection/mag_transition_autoencoder.py \
    src/detection/mag_residual_transition_autoencoder.py \
    src/data/mag_event_cache.py \
    scripts/4_preprocessing/4a_native_batches.py \
    scripts/5_training/5a_train_mag_cv.py
python -m unittest discover -s tests -v

echo "FIFTH MAG model: residual encoder (hidden=32, latent=8), 7 dilations."
echo "Same asymmetric before/after transition head as v4."
echo "All events, folds=1/2/3/4, epochs max=$EPOCHS min=$MIN_EPOCHS patience=$PATIENCE."
echo "Loss = L_rec + $LAMBDA_DETECTION*L_det + $LAMBDA_MAGNITUDE*L_mag."
echo "Gaussian targets/event=$TRAIN_WINDOWS_PER_EVENT; rotating samples/epoch=$TRAIN_WINDOWS_PER_EPOCH."
echo "Local raw-MAG cache: $MAG_CACHE_DIR (persisted across restarts; gitignored)."
echo "Cache refresh=$REFRESH_MAG_CACHE; BigQuery cap for cache MISSES=$MAX_TOTAL_GIB GiB."
echo "New model/checkpoints only in $OUTPUT_DIR."

mkdir -p "$OUTPUT_DIR"
extra_args=()
if [[ "$REFRESH_MAG_CACHE" == "1" ]]; then
    extra_args+=(--refresh-mag-cache)
elif [[ "$REFRESH_MAG_CACHE" != "0" ]]; then
    echo "REFRESH_MAG_CACHE must be 0 or 1." >&2
    exit 2
fi

python -u scripts/5_training/5a_train_mag_cv.py \
    --architecture residual-transition \
    --transition-side-offset-s "$TRANSITION_SIDE_OFFSET_S" \
    --transition-center-exclusion-s "$TRANSITION_CENTER_EXCLUSION_S" \
    --folds all \
    --epochs "$EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --early-stop-patience "$PATIENCE" \
    --train-windows-per-event "$TRAIN_WINDOWS_PER_EVENT" \
    --train-windows-per-epoch "$TRAIN_WINDOWS_PER_EPOCH" \
    --val-centers-per-event "$VAL_CENTERS_PER_EVENT" \
    --sigma-seconds "$SIGMA_SECONDS" \
    --lambda-detection "$LAMBDA_DETECTION" \
    --lambda-magnitude "$LAMBDA_MAGNITUDE" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    "${extra_args[@]}" \
    --max-train-events 0 \
    --max-val-events 0 \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --output-dir "$OUTPUT_DIR" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
