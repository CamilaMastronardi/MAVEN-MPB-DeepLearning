#!/usr/bin/env bash
# V12: rank multiple V10-refined MPB time hypotheses; V7b/V10 frozen.
# Requires completed V7b+V10 checkpoints for all selected folds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/metrics/mag_swea_multi_hypothesis_v12}"
V7B_DIR="${V7B_DIR:-results/metrics/mag_swea_regions_ranking_v1}"
V10_DIR="${V10_DIR:-results/metrics/mag_swea_refiner_v10}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
FOLDS="${FOLDS:-all}"
TOPK="${TOPK:-12}"
MIN_GAP_S="${MIN_GAP_S:-120}"
MAX_EPOCHS="${MAX_EPOCHS:-65}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVENTS_PER_UPDATE="${EVENTS_PER_UPDATE:-16}"
TORCH_THREADS="${TORCH_THREADS:-2}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"

echo "V12: frozen V7b proposals, TOPK=$TOPK diverse hypotheses"
echo "V10 frozen refinement of every hypothesis, trained permutation-invariant selector."
echo "Our goal is >=80% within +/-60s; V10 reference is 53.0%. Not guaranteed."
echo "Features cached individually per fold/event to avoid rescanning after interruption."
echo "All evaluations CONDITIONAL on a known MPB in each 70-min event block."
echo "Output: $OUTPUT_DIR"
if [[ "$PILOT_EVENTS_PER_FOLD" != "0" ]]; then
  echo "WARNING: small PILOT of $PILOT_EVENTS_PER_FOLD per fold."
  echo "PILOT METRICS are a functional smoke test, NOT a scientific comparison."
fi

python -m py_compile \
  src/detection/mpb_multi_hypothesis_v12.py \
  scripts/5_training/5h_train_mag_swea_multihypothesis_v12.py \
  tests/test_mag_swea_multihypothesis_v12.py

python -m unittest discover -s tests -p 'test_mag_swea_multihypothesis_v12.py' -v

mkdir -p "$OUTPUT_DIR"
python -u scripts/5_training/5h_train_mag_swea_multihypothesis_v12.py \
  --folds "$FOLDS" \
  --v7b-dir "$V7B_DIR" \
  --v10-dir "$V10_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --mag-cache-dir "$MAG_CACHE_DIR" \
  --swea-cache-dir "$SWEA_CACHE_DIR" \
  --topk "$TOPK" \
  --min-gap-s "$MIN_GAP_S" \
  --max-epochs "$MAX_EPOCHS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --events-per-update "$EVENTS_PER_UPDATE" \
  --torch-threads "$TORCH_THREADS" \
  --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
  --max-query-gib "$MAX_QUERY_GIB" \
  --max-total-gib "$MAX_TOTAL_GIB" \
  --resume --execute \
  2>&1 | tee -a "$OUTPUT_DIR/training.log"
