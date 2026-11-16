#!/usr/bin/env bash
# Full four-fold MAG cross-validation using ALL labeled events and real MAVEN data.
# Only model states, logs and summary metrics are saved; no science measurement arrays.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
fi

# Explicit controls for the full training run.
# MAX_TOTAL_GIB is a hard cap on SUM of BQ dry-run estimated processed bytes.
# It is NOT a runtime memory cap or a guaranteed monetary-cost ceiling.
EPOCHS="${EPOCHS:-1000}"
PATIENCE="${PATIENCE:-20}"
MIN_EPOCHS="${MIN_EPOCHS:-25}"
TRAIN_WINDOWS_PER_EVENT="${TRAIN_WINDOWS_PER_EVENT:-600}"
TRAIN_WINDOWS_PER_EPOCH="${TRAIN_WINDOWS_PER_EPOCH:-24}"
VAL_CENTERS_PER_EVENT="${VAL_CENTERS_PER_EVENT:-600}"
SIGMA_SECONDS="${SIGMA_SECONDS:-60}"
LAMBDA_DETECTION="${LAMBDA_DETECTION:-10}"
LAMBDA_MAGNITUDE="${LAMBDA_MAGNITUDE:-1}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_cv_600windows_det10_mag1_pat20_ep1000}"

echo "Checking MAG training syntax and offline tests before querying BigQuery..."
python -m py_compile src/detection/mag_autoencoder.py scripts/5_training/5a_train_mag_cv.py
python -m unittest discover -s tests -v

mkdir -p "$OUTPUT_DIR"
echo "MAG FULL cross-validation: ALL labeled TRAIN and VAL events, folds 1/2/3/4."
echo "max epochs=$EPOCHS, min epochs=$MIN_EPOCHS, patience=$PATIENCE"
echo "train Gaussian targets/event=$TRAIN_WINDOWS_PER_EVENT, per-epoch samples/event=$TRAIN_WINDOWS_PER_EPOCH"
echo "validation centers/event=$VAL_CENTERS_PER_EVENT, sigma=$SIGMA_SECONDS seconds, detection weight=$LAMBDA_DETECTION, magnitude weight=$LAMBDA_MAGNITUDE"
echo "max-total-gib=$MAX_TOTAL_GIB, output=$OUTPUT_DIR"
echo "BigQuery estimates are checked before any billable measurement query."

python -u scripts/5_training/5a_train_mag_cv.py \
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
    --max-train-events 0 \
    --max-val-events 0 \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --output-dir "$OUTPUT_DIR" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
