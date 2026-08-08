#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
DATA_DIR="$ROOT/corrected_production/data/processed_grids"
FOAM_DIR="$ROOT/corrected_production/openfoam_cases"
RUN_DIR="$ROOT/training_runs/corrected"
RESULT_DIR="$ROOT/paper_results/corrected"
OPT_DIR="$ROOT/hso_results/corrected/optimized"
TARGETS="Ux,Uy,p,nut,k,omega,Cp,cavitation_margin,Cl,Cd"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment not found at $PYTHON" >&2
  echo "Create it using the commands in REPRODUCIBILITY.md." >&2
  exit 1
fi

common_train_args=(
  --data-dir "$DATA_DIR"
  --output-dir "$RUN_DIR"
  --targets "$TARGETS"
  --epochs 120
  --early-stopping-patience 20
  --lr 0.002
  --weight-decay 0.0001
  --bce-weight 0.2
  --val-fraction 0.15
  --split-strategy naca
  --max-abs-force-coefficient 5
  --seed 7
)

run_data() {
  "$PYTHON" scripts/run_pipeline.py \
    --config configs/full_cases.yaml \
    --mode openfoam \
    --openfoam-backend docker \
    --run-cfd \
    --artifact-root corrected_production \
    --jobs "${JOBS:-8}" \
    --no-plots
  "$PYTHON" scripts/audit_cfd_dataset.py \
    --data-dir "$DATA_DIR" \
    --foam-dir "$FOAM_DIR" \
    --output "$RESULT_DIR/dataset_audit.json"
}

run_train() {
  "$PYTHON" -m models.train --model unet "${common_train_args[@]}" \
    --width 16 --depth 3 --operator-batch-size 4
  "$PYTHON" -m models.train --model pinn "${common_train_args[@]}" \
    --width 64 --depth 3 --batch-size 8192 --max-points-per-case 1024 \
    --physics-weight 0.0001
  "$PYTHON" -m models.train --model fno "${common_train_args[@]}" \
    --width 32 --depth 4 --modes 12 --operator-batch-size 4
  "$PYTHON" -m models.train --model deeponet "${common_train_args[@]}" \
    --width 64 --depth 3 --basis 64 --max-points-per-case 1024 \
    --batch-size 8192
}

run_evaluate() {
  "$PYTHON" scripts/evaluate_model_results.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" --output-dir "$RESULT_DIR" \
    --models all --split validation --split-strategy naca --seed 7
  "$PYTHON" scripts/evaluate_model_forces.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" --foam-dir "$FOAM_DIR" \
    --output-dir "$RESULT_DIR/model_forces" --split validation \
    --split-strategy naca --seed 7
  "$PYTHON" scripts/evaluate_cavitation_risk.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" \
    --output-dir "$RESULT_DIR/cavitation_risk" --split validation \
    --split-strategy naca --seed 7
  "$PYTHON" scripts/benchmark_runtime_speedup.py \
    --run-dir "$RUN_DIR" --data-dir "$DATA_DIR" --openfoam-dir "$FOAM_DIR" \
    --output "$RESULT_DIR/runtime_speedup.csv" --split validation --seed 7
  "$PYTHON" scripts/plot_training_histories.py \
    --run-dir "$RUN_DIR" --output "$RESULT_DIR/training_loss_curves.png"
}

run_optimize() {
  "$PYTHON" scripts/optimize_hydrofoil_shapes.py \
    --run-dir "$RUN_DIR" --models unet,pinn,fno,deeponet \
    --output-dir "$OPT_DIR" --Re 500000 --maxiter 40
  "$PYTHON" scripts/validate_optimized_designs.py \
    --optimization-summary "$OPT_DIR/optimization_summary.csv" \
    --artifact-root "$ROOT/hso_results/corrected/openfoam_validation" \
    --output "$OPT_DIR/openfoam_validation.csv" --jobs "${JOBS:-4}" --Re 500000
}

run_papers() {
  local paper
  for paper in ai4s26 sim2science26; do
    (cd "$ROOT/papers/$paper" && \
      TECTONIC_CACHE_DIR="$ROOT/.cache/tectonic" tectonic main.tex \
        --keep-logs --keep-intermediates)
  done
}

case "$STAGE" in
  data) run_data ;;
  train) run_train ;;
  evaluate) run_evaluate ;;
  optimize) run_optimize ;;
  papers) run_papers ;;
  all)
    run_data
    run_train
    run_evaluate
    run_optimize
    run_papers
    ;;
  *)
    echo "Usage: $0 {data|train|evaluate|optimize|papers|all}" >&2
    exit 2
    ;;
esac
