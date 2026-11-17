#!/usr/bin/env bash
# Controlled V7b region-weight sweep: only --region-weight varies; rank_weight=2 and rank_margin=0.25.
# All runs share source caches and fixed fold/seed/hyperparameters.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/metrics/v7b_region_weight_sweep}"
FOLDS="${FOLDS:-all}"
PILOT_EVENTS_PER_FOLD="${PILOT_EVENTS_PER_FOLD:-0}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
MIN_EPOCHS="${MIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0003}"
RANK_MARGIN="0.25"
RECONSTRUCTION_WEIGHT="${RECONSTRUCTION_WEIGHT:-0.15}"
RANK_WEIGHT="2"
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
echo "V7b controlled region-weight sweep: 0.25, 0.5, 1, 2 (rank weight 2; margin 0.25)"
echo "Folds=$FOLDS; pilot_events_per_fold=$PILOT_EVENTS_PER_FOLD; seed=$SEED"
echo "Outputs: $OUTPUT_ROOT"
echo "All region weights use the same fit/monitor partitions; checkpoints chosen on inner monitor."
echo "Outer-fold errors are for reporting, never checkpoint selection."
python -m py_compile \
  scripts/02_training/01_train_v7b.py \
  src/model/v7b.py \
  src/model/ranking.py \
  scripts/03_experiments/03_compare_region_weight.py
python -m unittest discover -s tests -q


# Reuse the previous rank_weight=2, rank_margin=0.25, region_weight=0.5 full CV run.
# Fail before training if its configuration differs from this experiment.
REUSE_BASELINE="${REUSE_BASELINE:-1}"
BASELINE_DIR="${BASELINE_DIR:-results/metrics/v7b_ranking_margin_sweep/margin_0p25}"
if [[ "$REUSE_BASELINE" == "1" && "$FOLDS" == "all" && "$PILOT_EVENTS_PER_FOLD" == "0" ]]; then
  python - "$BASELINE_DIR" "$MANIFEST" "$SEED" "$MAX_EPOCHS" "$MIN_EPOCHS" "$PATIENCE" "$BATCH_SIZE" "$LEARNING_RATE" "$RANK_MARGIN" "$RECONSTRUCTION_WEIGHT" "$EVALUATION_STRIDE_S" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
root, manifest = Path(sys.argv[1]), Path(sys.argv[2])
expected = {
    "seed": int(sys.argv[3]),
    "max_epochs": int(sys.argv[4]),
    "epochs_min": int(sys.argv[5]),
    "patience": int(sys.argv[6]),
    "batch_size": int(sys.argv[7]),
    "learning_rate": float(sys.argv[8]),
    "rank_margin": float(sys.argv[9]),
    "region_weight": 0.5,
    "reconstruction_weight": float(sys.argv[10]),
    "evaluation_stride_s": float(sys.argv[11]),
    "rank_weight": 2.0,

    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
}
summary_path = root / "cv_summary.json"
if not summary_path.exists():
    sys.exit(f"Falta la referencia V7b: {summary_path}. Usar REUSE_BASELINE=0 para entrenarla de nuevo.")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("pilot_subset") or sorted(summary.get("completed_folds", [])) != [1,2,3,4]:
    sys.exit("La referencia V7b no contiene los cuatro folds completos.")
config = summary.get("training_configuration", {})
for k,v in expected.items():
    if k not in config or config[k] != v:
        sys.exit(f"Configuracion incompatible de la referencia: {k}: {config.get(k)!r} != {v!r}")
for fold in range(1,5):
    f = root / f"fold_{fold:02d}_metrics.json"
    pt = root / f"fold_{fold:02d}_final.pt"
    csv = root / f"fold_{fold:02d}_event_metrics.csv"
    if not (f.is_file() and pt.is_file() and csv.is_file()):
        sys.exit(f"Falta un resultado de la referencia fold {fold}.")
    metric = json.loads(f.read_text(encoding="utf-8"))
    if (metric["training_configuration"]["rank_weight"] != 2.0 or metric["training_configuration"]["rank_margin"] != 0.25 or metric["training_configuration"]["region_weight"] != 0.5):
        sys.exit(f"Ranking incorrecto en referencia fold {fold}.")
print(f"Referencia previa validada: {root}")
PY
  if [[ -L "$OUTPUT_ROOT/region_0p5" ]]; then
    if [[ "$(realpath "$OUTPUT_ROOT/region_0p5")" != "$(realpath "$BASELINE_DIR")" ]]; then
      echo "El enlace region_0p5 apunta a otra ejecucion." >&2; exit 1
    fi
  elif [[ -e "$OUTPUT_ROOT/region_0p5" ]]; then
    echo "region_0p5 ya existe como carpeta independiente; se respetara." 
  else
    ln -s "$(realpath "$BASELINE_DIR")" "$OUTPUT_ROOT/region_0p5"
    echo "Reutilizando 4 folds con ranking 2, margen 0.25 y peso de regiones 0.5."
  fi
elif [[ "$REUSE_BASELINE" != "0" && "$REUSE_BASELINE" != "1" ]]; then
  echo "REUSE_BASELINE debe ser 0 o 1." >&2; exit 2
fi

# Sequential, not parallel: avoid duplicate BigQuery fetches and CPU contention.
# --resume checks the original V7b signature including rank weight and fold.
# Each region weight has its own best/last/final checkpoints and CV metrics.
for spec in '0.25:region_0p25' '0.5:region_0p5' '1:region_1' '2:region_2'; do
  region_weight="${spec%%:*}"
  dirname="${spec#*:}"
  out="$OUTPUT_ROOT/$dirname"
  if [[ "$region_weight" == "0.5" && -L "$out" ]]; then
    echo "Skipping region weight 0.5: reuse from previous margin sweep."
    continue
  fi
  mkdir -p "$out"
  echo "============================================================"
  echo "Training V7b region weight $region_weight (rank_weight=2, margin=0.25) in $out"
  echo "============================================================"
  python -u scripts/02_training/01_train_v7b.py \
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
    --region-weight "$region_weight" \
    --reconstruction-weight "$RECONSTRUCTION_WEIGHT" \
    --rank-weight "$RANK_WEIGHT" \
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

# Analyze *matched* event identifiers across region weights; select each fold's region weight
# using ONLY its inner monitor top-1 ranking, never outer validation.
python scripts/03_experiments/03_compare_region_weight.py \
  --results-root "$OUTPUT_ROOT" --folds "$FOLDS"