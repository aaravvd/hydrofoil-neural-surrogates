#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAGE="${1:-all}"
DATA_DIR="$ROOT/corrected_production/data/processed_grids"
FOAM_DIR="$ROOT/corrected_production/openfoam_cases"
RUN_DIR="$ROOT/training_runs/revised"
RESULT_DIR="$ROOT/paper_results/revised"
OPT_DIR="$ROOT/hso_results/revised"
CFD_OPT_DIR="$ROOT/hso_results/corrected/openfoam_baseline"
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
  --split-manifest "$ROOT/configs/family_split.json"
  --max-abs-force-coefficient 5
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
  local seed seed_dir
  for seed in ${SEEDS:-7 17 27}; do
    seed_dir="$RUN_DIR/seed_$seed"
    "$PYTHON" -m models.train --model unet "${common_train_args[@]}" --output-dir "$seed_dir" --seed "$seed" \
      --width 16 --depth 3 --operator-batch-size 4
    "$PYTHON" -m models.train --model pinn "${common_train_args[@]}" --output-dir "$seed_dir" --seed "$seed" \
      --width 64 --depth 3 --batch-size 8192 --max-points-per-case 1024 \
      --physics-weight 0.0001
    "$PYTHON" -m models.train --model fno "${common_train_args[@]}" --output-dir "$seed_dir" --seed "$seed" \
      --width 32 --depth 4 --modes 12 --operator-batch-size 4
    "$PYTHON" -m models.train --model deeponet "${common_train_args[@]}" --output-dir "$seed_dir" --seed "$seed" \
      --width 64 --depth 3 --basis 64 --max-points-per-case 1024 \
      --batch-size 8192
  done
}

run_evaluate() {
  local seed seed_run seed_result
  for seed in ${SEEDS:-7 17 27}; do
    seed_run="$RUN_DIR/seed_$seed"
    seed_result="$RESULT_DIR/seed_$seed"
    "$PYTHON" scripts/evaluate_model_results.py \
      --run-dir "$seed_run" --data-dir "$DATA_DIR" --output-dir "$seed_result" \
      --models all --split test --split-manifest "$ROOT/configs/family_split.json" --seed "$seed"
    "$PYTHON" scripts/evaluate_model_forces.py \
      --run-dir "$seed_run" --data-dir "$DATA_DIR" --foam-dir "$FOAM_DIR" \
      --output-dir "$seed_result/model_forces" --split test \
      --split-manifest "$ROOT/configs/family_split.json" --seed "$seed"
    "$PYTHON" scripts/evaluate_cavitation_risk.py \
      --run-dir "$seed_run" --data-dir "$DATA_DIR" \
      --output-dir "$seed_result/cavitation_risk" --split test \
      --split-manifest "$ROOT/configs/family_split.json" --seed "$seed"
    "$PYTHON" scripts/benchmark_runtime_speedup.py \
      --run-dir "$seed_run" --data-dir "$DATA_DIR" --openfoam-dir "$FOAM_DIR" \
      --output "$seed_result/runtime_speedup.csv" --split test \
      --split-manifest "$ROOT/configs/family_split.json" --seed "$seed"
    "$PYTHON" scripts/plot_training_histories.py \
      --run-dir "$seed_run" --output "$seed_result/training_loss_curves.png"
  done
  "$PYTHON" scripts/aggregate_multiseed_results.py --result-dir "$RESULT_DIR" --run-dir "$RUN_DIR"
  "$PYTHON" scripts/visualize_model_predictions.py \
    --run-dir "$RUN_DIR/seed_7" --data-dir "$DATA_DIR" --field p \
    --case case_179 --num-cases 1 --output-dir "$ROOT/figures/revised_model_predictions_test"
}

run_optimize() {
  local seed seed_opt
  for seed in ${SEEDS:-7 17 27}; do
    seed_opt="$OPT_DIR/seed_$seed"
    "$PYTHON" scripts/optimize_hydrofoil_shapes.py \
      --run-dir "$RUN_DIR/seed_$seed" --models unet,pinn,fno,deeponet \
      --output-dir "$seed_opt" --Re 500000 --maxiter 40 --cavitation-mode diagnostic
    "$PYTHON" scripts/validate_optimized_designs.py \
      --optimization-summary "$seed_opt/optimization_summary.csv" \
      --artifact-root "$OPT_DIR/openfoam_validation/seed_$seed" \
      --output "$seed_opt/openfoam_validation.csv" --jobs "${JOBS:-4}" --Re 500000
  done
  "$PYTHON" scripts/aggregate_hso_multiseed.py --input-dir "$OPT_DIR"
  "$PYTHON" scripts/plot_optimization_traces.py --input-dir "$OPT_DIR" --output "$OPT_DIR/optimization_traces.png"
  if [[ "${RUN_DIRECT_CFD_BASELINE:-0}" == "1" ]]; then
    "$PYTHON" scripts/optimize_openfoam_baseline.py \
      --output-dir "$CFD_OPT_DIR" --starts 0012,2412,4415 \
      --Re 500000 --maxiter 40
  fi
  "$PYTHON" scripts/grid_convergence_selected_designs.py \
    --optimization-summary "$OPT_DIR/seed_7/optimization_summary.csv" \
    --output-dir "$OPT_DIR/grid_study" --jobs "${JOBS:-4}" --Re 500000
}

run_papers() {
  local paper
  for paper in ai4s26 sim2science26; do
    (cd "$ROOT/papers/$paper" && \
      TECTONIC_CACHE_DIR="$ROOT/.cache/tectonic" tectonic main.tex \
        --keep-logs --keep-intermediates)
  done
  mkdir -p "$ROOT/output/pdf"
  cp "$ROOT/papers/ai4s26/main.pdf" \
    "$ROOT/output/pdf/ai4s26_hydrofoil_surrogate_optimization_draft.pdf"
  cp "$ROOT/papers/sim2science26/main.pdf" \
    "$ROOT/output/pdf/sim2science26_imperfect_cfd_hydrofoil_draft.pdf"
  mkdir -p "$ROOT/output/pdf/ai4s26" "$ROOT/output/pdf/sim2science26"
  cp "$ROOT/papers/ai4s26/main.pdf" \
    "$ROOT/output/pdf/ai4s26/hydrofoil_surrogate_benchmark_ai4s26.pdf"
  cp "$ROOT/papers/sim2science26/main.pdf" \
    "$ROOT/output/pdf/sim2science26/hydrofoil_surrogate_benchmark_sim2science26.pdf"
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
