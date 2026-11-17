#!/usr/bin/env bash
# V13 interpretation: physics/latents/trajectories, no retraining or BigQuery.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/v13_latent_physics_trajectories}"
MAG_DIR="${MAG_DIR:-results/metrics/mag_autoencoder_clusters_v13}"
SWEA_DIR="${SWEA_DIR:-results/metrics/swea_mag_joint_clusters_v13_frames256}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
EVENTS_PER_FOLD="${EVENTS_PER_FOLD:-0}"
PLOT_EXAMPLES="${PLOT_EXAMPLES:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TORCH_THREADS="${TORCH_THREADS:-2}"

echo "V13 MAG/SWEA latent physics: measured pre/post MPB features."
echo "Scan 10-min candidate centers -240 to +240s by 30s."
echo "No model training; no BigQuery or data download."
echo "Fold 4 kept separate from FIT physical correlations."
echo "Output: $OUTPUT_DIR"

test -f "$MAG_DIR/discovery_final.pt" || {
  echo "Missing MAG V13 encoder: $MAG_DIR/discovery_final.pt" >&2
  exit 1
}
test -f "$SWEA_DIR/discovery_final.pt" || {
  echo "Missing SWEA V13 encoder: $SWEA_DIR/discovery_final.pt" >&2
  exit 1
}

python -m py_compile \
  src/analysis/mpb_latent_physics_v13.py \
  scripts/6_latent_analysis/6a_latent_physics_trajectories_v13.py \
  tests/test_latent_physics_trajectories_v13.py

python -m unittest discover \
  -s tests -p 'test_latent_physics_trajectories_v13.py' -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/6_latent_analysis/6a_latent_physics_trajectories_v13.py \
  --mag-dir "$MAG_DIR" \
  --swea-dir "$SWEA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --events-per-fold "$EVENTS_PER_FOLD" \
  --plot-examples "$PLOT_EXAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --torch-threads "$TORCH_THREADS" \
  --execute 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"
