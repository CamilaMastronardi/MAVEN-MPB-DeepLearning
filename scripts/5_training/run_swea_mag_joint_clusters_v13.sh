#!/usr/bin/env bash
# V13 SWEA: independent 8-layer 3D AE, then MAG+SWEA clustering in 6D.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/swea_mag_joint_clusters_v13_frames256}"
MAG_LATENTS_CSV="${MAG_LATENTS_CSV:-results/metrics/mag_autoencoder_clusters_v13/crossing_latents_groups.csv}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
DISCOVERY_HOLDOUT_FOLD="${DISCOVERY_HOLDOUT_FOLD:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-65}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TORCH_THREADS="${TORCH_THREADS:-2}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V13 SWEA AE: 12 real-energy channels, native SWEA survey frames."
echo "8 CNN encoder layers, three scalar latent variables, 8 CNN decoder layers."
echo "MAG 3D embeddings REUSED from: $MAG_LATENTS_CSV"
echo "SWEA-only KMeans in 3D and MAG+SWEA KMeans in standardized 6D."
echo "PCA 3D visualization does NOT determine clusters."
echo "All preprocessing, KMeans and PCA fit only on FIT TRAIN crossing events."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
  echo "PILOT: functional smoke test, not scientifically comparable grouping."
fi

test -f "$MAG_LATENTS_CSV" || {
  echo "Missing prior MAG crossing latent CSV: $MAG_LATENTS_CSV" >&2
  exit 1
}

python -m py_compile \
  src/detection/mpb_swea_autoencoder_v13.py \
  scripts/5_training/5j_train_swea_autoencoder_joint_v13.py \
  tests/test_swea_mag_joint_clusters_v13.py

python -m unittest discover \
  -s tests -p 'test_swea_mag_joint_clusters_v13.py' -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5j_train_swea_autoencoder_joint_v13.py \
  --mag-latents-csv "$MAG_LATENTS_CSV" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --discovery-holdout-fold "$DISCOVERY_HOLDOUT_FOLD" \
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
