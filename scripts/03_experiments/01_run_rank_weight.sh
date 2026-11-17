#!/usr/bin/env bash
# Controlled V7b weight sweep: only --rank-weight varies.
# All runs share source caches and fixed fold/seed/hyperparameters.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/metrics/v7b_ranking_weight_sweep}"
FOLDS="${FOLDS:-all}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
REGION_WEIGHT="${REGION_WEIGHT:-0.5}"
RECONSTRUCTION_WEIGHT="${RECONSTRUCTION_WEIGHT:-0.15}"
RANK_MARGIN="${RANK_MARGIN:-0.5}"
RANK_CONTROL_STEP_S="${RANK_CONTROL_STEP_S:-120}"
RANK_POSITIVES_PER_EVENT="${RANK_POSITIVES_PER_EVENT:-2}"
RANK_HARD_NEGATIVES="${RANK_HARD_NEGATIVES:-2}"
RANK_RANDOM_NEGATIVES="${RANK_RANDOM_NEGATIVES:-2}"
EVALUATION_STRIDE_S="${EVALUATION_STRIDE_S:-32}"
TORCH_THREADS="${TORCH_THREADS:-2}"
MAX_QUERY_GIB="${MAX_QUERY_GIB:-2}"
MAX_TOTAL_GIB="${MAX_TOTAL_GIB:-150}"
MAG_CACHE_DIR="${MAG_CACHE_DIR:-data/cache/mag_events}"
SWEA_CACHE_DIR="${SWEA_CACHE_DIR:-data/cache/swea_events}"
MANIFEST="${MANIFEST:-data/splits/cv4_monthly_manifest.csv}"

case "$FOLDS" in all|1|2|3|4) ;; *) echo "FOLDS must be all, 1, 2, 3 or 4" >&2; exit 2;; esac

mkdir -p "$OUTPUT_ROOT"
echo "V7b controlled ranking-weight sweep: 0.5, 1, 2, 4"
echo "Folds=$FOLDS; pilot_events_per_fold=$PILOT_EVENTS_PER_FOLD; seed=$SEED"
echo "Outputs: $OUTPUT_ROOT"
echo "All weights use the same fit/monitor month partitions; checkpoints chosen on inner monitor."
echo "Outer-fold errors are for reporting, never checkpoint selection."
python -m py_compile \
  scripts/5_training/5b_train_mag_swea_regions.py \
  src/detection/mag_swea_region_model.py \
  src/detection/mpb_event_ranking.py \
  scripts/5_training/compare_v7b_rank_weight_sweep.py
python -m unittest discover -s tests -q

# Sequential, not parallel: avoid duplicate BigQuery fetches and CPU contention.
# --resume checks the original V7b signature including rank weight and fold.
# Each weight has its own best/last/final checkpoints and CV metrics.
for spec in '0.5:rank_0p5' '1:rank_1' '2:rank_2' '4:rank_4'; do
  rank_weight="${spec%%:*}"
  dirname="${spec#*:}"
  out="$OUTPUT_ROOT/$dirname"
  mkdir -p "$out"
  echo "============================================================"
  echo "Training V7b rank weight $rank_weight in $out"
  echo "============================================================"
  python -u scripts/5_training/5b_train_mag_swea_regions.py \
    --manifest "$MANIFEST" \
    --output-dir "$out" \
    --mag-cache-dir "$MAG_CACHE_DIR" \
    --swea-cache-dir "$SWEA_CACHE_DIR" \
    --folds "$FOLDS" \
    --pilot-events-per-fold "$PILOT_EVENTS_PER_FOLD" \
    --seed "$SEED" \
    --max-epochs "$MAX_EPOCHS" \
    --min-epochs "$MIN_EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --region-weight "$REGION_WEIGHT" \
    --reconstruction-weight "$RECONSTRUCTION_WEIGHT" \
    --rank-weight "$rank_weight" \
    --rank-margin "$RANK_MARGIN" \
    --rank-control-step-s "$RANK_CONTROL_STEP_S" \
    --rank-positives-per-event "$RANK_POSITIVES_PER_EVENT" \
    --rank-hard-negatives "$RANK_HARD_NEGATIVES" \
    --rank-random-negatives "$RANK_RANDOM_NEGATIVES" \
    --evaluation-stride-s "$EVALUATION_STRIDE_S" \
    --torch-threads "$TORCH_THREADS" \
    --max-query-gib "$MAX_QUERY_GIB" \
    --max-total-gib "$MAX_TOTAL_GIB" \
    --resume --execute 2>&1 | tee -a "$out/training.log"
done

# Analyze *matched* event identifiers across weights; select each fold's weight
# using ONLY its inner monitor top-1 ranking, never outer validation.
python scripts/5_training/compare_v7b_rank_weight_sweep.py \
  --results-root "$OUTPUT_ROOT" --folds "$FOLDS"