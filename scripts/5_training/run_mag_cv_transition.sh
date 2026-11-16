#!/usr/bin/env bash
# Fourth full MAG experiment: asymmetric BEFORE/CENTER/AFTER/DELTA detector.
# Does NOT change or re-use the running third experiment's checkpoints.
# All events and all 4 monthly held-out folds, no scientific mock data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
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
OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_cv_transition_v1}"

echo "Checking transition model and offline tests before any BigQuery query..."
python -m py_compile \
    src/detection/mag_autoencoder.py \
    src/detection/mag_transition_autoencoder.py \
    scripts/4_preprocessing/4a_native_batches.py \
    scripts/5_training/5a_train_mag_cv.py
python -m unittest discover -s tests -v

echo "Fourth MAG run: independent asymmetric transition detector."
echo "4 folds; 569 known events; epochs max=$EPOCHS, min=$MIN_EPOCHS, patience=$PATIENCE"
echo "Transition offsets +/-$TRANSITION_SIDE_OFFSET_S s; central exclusion +/-$TRANSITION_CENTER_EXCLUSION_S s"
echo "Loss: L_rec + ${LAMBDA_DETECTION}*L_det + ${LAMBDA_MAGNITUDE}*L_mag"
echo "600 Gaussian targets/event, $TRAIN_WINDOWS_PER_EPOCH rotating train samples/epoch"
echo "BigQuery estimate cap=$MAX_TOTAL_GIB GiB; output=$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5a_train_mag_cv.py \
    --architecture transition \
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
    --max-train-events 0 \
    --max-val-events 0 \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --output-dir "$OUTPUT_DIR" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
