#!/usr/bin/env bash
# Four-fold MAG real-data pilot. No artificial training data or saved measurements.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

echo "Checking syntax and offline tests before BigQuery work..."
python -m py_compile src/detection/mag_autoencoder.py scripts/5_training/5a_train_mag_cv.py
python -m unittest discover -s tests -v

mkdir -p results/metrics/mag_cv
echo "Starting REAL MAVEN MAG four-fold SUBSET pilot."
echo "Selection: 30 TRAIN + 10 VAL per fold; 3 epochs; 35 GiB estimated cap."
echo "NOT full 569-event cross-validation."
python -u scripts/5_training/5a_train_mag_cv.py \
  --folds all --epochs 3 \
  --max-train-events 30 --max-val-events 10 \
  --max-total-gib 35 --resume --execute \
  2>&1 | tee -a results/metrics/mag_cv/training.log
