#!/usr/bin/env bash
# Sixth FULL MAG experiment: v5 residual encoder, dense temporal transition
# score for EVERY native MAG epoch, early stopping min=15 patience=10.
# Independent results and checkpoint signatures; never touches v5 weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

EPOCHS="${EPOCHS:-1000}"
MIN_EPOCHS="${MIN_EPOCHS:-15}"
PATIENCE="${PATIENCE:-10}"
TRAIN_WINDOWS_PER_EVENT="${TRAIN_WINDOWS_PER_EVENT:-600}"
# 40 rotating centers x 15 epochs = all 600 unique centers by epoch 15.
TRAIN_WINDOWS_PER_EPOCH="${TRAIN_WINDOWS_PER_EPOCH:-40}"
VAL_CENTERS_PER_EVENT="${VAL_CENTERS_PER_EVENT:-600}"
SIGMA_SECONDS="${SIGMA_SECONDS:-60}"
LAMBDA_DETECTION="${LAMBDA_DETECTION:-10}"
LAMBDA_MAGNITUDE="${LAMBDA_MAGNITUDE:-1}"
TRANSITION_SIDE_OFFSET_S="${TRANSITION_SIDE_OFFSET_S:-120}"
TRANSITION_CENTER_EXCLUSION_S="${TRANSITION_CENTER_EXCLUSION_S:-15}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_cv_dense_temporal_v1}"
REFRESH_MAG_CACHE="${REFRESH_MAG_CACHE:-0}"

echo "Sixth MAG experiment: checking native timestamp, dense scoring and cache tests."
python -m py_compile \
  src/detection/mag_dense_temporal_autoencoder.py \
  src/detection/mag_temporal_scoring.py \
  scripts/5_training/5a_train_mag_cv.py \
  scripts/6_figures/6b_plot_mag_reconstruction_cases.py
python -m unittest discover -s tests -v

echo "V6 MAG: residual hidden=32, latent=8, 7 dilations; dense native-time head."
echo "Early-stop min_epochs=$MIN_EPOCHS, patience=$PATIENCE; max_epochs=$EPOCHS."
echo "Training centers/event=$TRAIN_WINDOWS_PER_EVENT; per epoch=$TRAIN_WINDOWS_PER_EPOCH."
echo "Full 569-event four-fold CV; 600 evaluation window centers/event."
echo "Validation: temporal scores averaged on original MAG timestamps in +/-30min."
echo "Loss=L_rec+ $LAMBDA_DETECTION*L_dense_det + $LAMBDA_MAGNITUDE*L_mag."
echo "Cache=$MAG_CACHE_DIR; output=$OUTPUT_DIR."

mkdir -p "$OUTPUT_DIR"
extra=()
if [[ "$REFRESH_MAG_CACHE" == "1" ]]; then
  extra+=(--refresh-mag-cache)
elif [[ "$REFRESH_MAG_CACHE" != "0" ]]; then
  echo "REFRESH_MAG_CACHE must be 0 or 1" >&2
  exit 2
fi

python -u scripts/5_training/5a_train_mag_cv.py \
  --architecture dense-temporal \
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
  "${extra[@]}" \
  --max-train-events 0 \
  --max-val-events 0 \
  --max-total-gib "$MAX_TOTAL_GIB" \
  --output-dir "$OUTPUT_DIR" \
  --resume --execute \
  2>&1 | tee -a "$OUTPUT_DIR/training.log"
