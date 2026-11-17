#!/usr/bin/env bash
# V13 phase 1: 8-layer MAG CNN autoencoder, 3D latent, crossing clusters.
# Run smoke test first; no full-event 60-s localization claim yet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_autoencoder_clusters_v13}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
DISCOVERY_HOLDOUT_FOLD="${DISCOVERY_HOLDOUT_FOLD:-4}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-65}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TORCH_THREADS="${TORCH_THREADS:-2}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V13 STAGE 1: MAG measured 10-min windows, 8-layer CNN encoder,"
echo "three-dimensional latent bottleneck, 8-layer decoder."
echo "Main loss: masked MAG reconstruction; auxiliary MPB-window BCE"
echo "and positive-window time-offset regression."
echo "KMeans groups learned from FIT crossing embeddings, remaining events assigned."
echo "No temporal localization >=80% claim in this DISCOVERY phase."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
  echo "WARNING: pilot subset. Small clusters may be unstable."
fi

python -m py_compile \
  src/detection/mpb_cnn_autoencoder_v13.py \
  scripts/5_training/5i_train_mag_autoencoder_clusters_v13.py \
  tests/test_mag_autoencoder_clusters_v13.py

python -m unittest discover \
  -s tests -p 'test_mag_autoencoder_clusters_v13.py' -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5i_train_mag_autoencoder_clusters_v13.py \
  --discovery-holdout-fold "$DISCOVERY_HOLDOUT_FOLD" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
  --max-epochs "$MAX_EPOCHS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --torch-threads "$TORCH_THREADS" \
  --max-query-gib "$MAX_QUERY_GIB" \
  --max-total-gib "$MAX_TOTAL_GIB" \
  --resume --execute \
  2>&1 | tee -a "$OUTPUT_DIR/training.log"
