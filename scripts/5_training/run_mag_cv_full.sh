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
EPOCHS="${EPOCHS:-10}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_cv_full}"

echo "Checking MAG training syntax and offline tests before querying BigQuery..."
python -m py_compile src/detection/mag_autoencoder.py scripts/5_training/5a_train_mag_cv.py
python -m unittest discover -s tests -v

mkdir -p "$OUTPUT_DIR"
echo "MAG FULL cross-validation: ALL labeled TRAIN and VAL events, folds 1/2/3/4."
echo "epochs=$EPOCHS, max-total-gib=$MAX_TOTAL_GIB, output=$OUTPUT_DIR"
echo "BigQuery estimates are checked before any billable measurement query."

python -u scripts/5_training/5a_train_mag_cv.py \
    --folds all \
    --epochs "$EPOCHS" \
    --max-train-events 0 \
    --max-val-events 0 \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --output-dir "$OUTPUT_DIR" \
    --resume --execute \
    2>&1 | tee -a "$OUTPUT_DIR/training.log"
