# Hydrofoil Neural Surrogate Benchmark

This repository benchmarks CNN-U-Net, a physics-regularized point network,
FNO, and DeepONet on hydrofoil RANS flow prediction, force prediction,
cavitation-risk screening, runtime, and hydrodynamic shape optimization.

The code is released under the [MIT License](LICENSE). See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact paper workflow,
environment, output locations, and claim boundaries.

Dataset and trained checkpoints: [Zenodo DOI 10.5281/zenodo.21845241](https://doi.org/10.5281/zenodo.21845241).

The local workflow supports two modes:

- `analytic_smoke`: deterministic synthetic fields used to verify the full data
  path for 5 cases.
- `openfoam`: RANS cases executed with host OpenFOAM or the local Docker image.

All results reported in the AI4S’26 paper were generated using the corrected CFD dataset archived at [(https://doi.org/10.5281/zenodo.21845241)]. The earlier pre-release dataset affected by an angle-of-attack boundary-condition error was not used for any results in the paper and is not included in the archived artifact.

## Quick Start: 5-Case Pilot

```bash
python3 scripts/run_pipeline.py --config configs/pilot_cases.yaml --mode analytic_smoke
```

This creates:

```text
data/raw_cases/case_001.npz
data/processed_grids/case_001_grid.npz
data/metadata.csv
figures/case_001_pressure.png
figures/case_001_velocity.png
figures/case_001_cavitation.png
openfoam_cases/case_001/
```

## Scaling

Generate a larger manifest:

```bash
python3 scripts/make_manifest.py --preset full --output configs/full_cases.yaml
python3 scripts/run_pipeline.py --config configs/full_cases.yaml --mode analytic_smoke
```

When OpenFOAM is available through the macOS Docker wrapper/image:

```bash
python3 scripts/run_pipeline.py --config configs/full_cases.yaml --mode openfoam \
  --openfoam-backend docker --run-cfd --artifact-root corrected_production \
  --jobs 8 --no-plots
```

## Saved Variables

Each raw case `.npz` contains:

```text
x, y
Ux, Uy, p
nut, k, omega
Rxx, Rxy, Ryy
Cp, cavitation_margin, cavitation_indicator
airfoil_x, airfoil_y
Re, AoA, U_inf, rho, nu, p_inf, p_vap
```

Each processed grid `.npz` contains the same field variables interpolated onto
a common Cartesian grid, plus `fluid_mask`, signed distance (`sdf`), global
shape descriptors, and direct OpenFOAM `Cl`/`Cd` targets.

## Notes

- Cavitation labels use absolute pressure: `p_abs < p_vap`.
- The pilot Re/chord settings produce low water speeds, so the physical
  `cavitation_indicator` may be all zero. Use `cavitation_margin` and `Cp` for
  risk ranking, or raise speed/lower ambient pressure in a cavitation-focused
  sweep.
- OpenFOAM pressure may be gauge pressure depending on the case setup. The
  pipeline stores `p_inf` so post-processing can convert consistently.
- Reynolds stresses are modeled RANS stresses, computed from `k`, `nut`, and
  velocity gradients in pilot mode.

## Exact Paper Workflow

Create an environment with the package versions used for the reported runs:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
```

Run one stage at a time or reproduce the complete experiment:

```bash
./scripts/reproduce_paper.sh data
./scripts/reproduce_paper.sh train
./scripts/reproduce_paper.sh evaluate
./scripts/reproduce_paper.sh optimize
./scripts/reproduce_paper.sh papers

# Or run all five stages in order.
./scripts/reproduce_paper.sh all
```

The stage runner uses the exact paper targets and checkpoint hyperparameters.
It trains for at most 120 epochs with patience 20 using seeds 7, 17, and 27.
The fixed family manifest reserves NACA 0015 and 2412 for validation and NACA
0018 and 4418 as a final test set that is never used for fitting, early
stopping, or model selection.

The physics-regularized model uses supervised CFD targets plus a lightweight
steady incompressible continuity and momentum loss. It is not a data-free PINN
or a complete RANS residual.

After training, plot pressure predictions on validation cases:

```bash
.venv/bin/python scripts/evaluate_model_results.py \
  --run-dir training_runs/revised/seed_7 \
  --data-dir corrected_production/data/processed_grids \
  --output-dir paper_results/revised/seed_7 --split test

.venv/bin/python scripts/visualize_model_predictions.py \
  --run-dir training_runs/revised/seed_7 \
  --data-dir corrected_production/data/processed_grids --field p --num-cases 3
```

This writes comparison figures to `figures/model_predictions/`, with truth,
each available model prediction, and absolute-error maps.

Benchmark surrogate inference against existing OpenFOAM logs:

```bash
.venv/bin/python scripts/benchmark_runtime_speedup.py \
  --run-dir training_runs/revised/seed_7 \
  --data-dir corrected_production/data/processed_grids \
  --openfoam-dir corrected_production/openfoam_cases \
  --output paper_results/revised/seed_7/runtime_speedup.csv --split test
```

Run surrogate-assisted hydrodynamic shape optimization/screening over NACA
foils and randomly generated smooth airfoils:

```bash
.venv/bin/python scripts/hydrodynamic_shape_optimization.py \
  --run-dir training_runs/corrected --models unet,pinn,fno,deeponet \
  --n-random-airfoils 5 --output-dir hso_results/corrected/screening

.venv/bin/python scripts/optimize_hydrofoil_shapes.py \
  --run-dir training_runs/corrected --models unet,pinn,fno,deeponet \
  --output-dir hso_results/corrected/optimized

.venv/bin/python scripts/optimize_openfoam_baseline.py \
  --output-dir hso_results/corrected/openfoam_baseline \
  --starts 0012,2412,4415 --Re 500000 --maxiter 40
```

This writes ranked design candidates, candidate geometry CSVs, and a screening
report under `hso_results/`. Corrected checkpoints rank with learned OpenFOAM
`Cl`/`Cd` targets. Each surrogate winner is remeshed and rerun once in OpenFOAM
to verify its endpoint, while `optimize_openfoam_baseline.py` performs the
matched direct-CFD search from the same starts, with the same bounds, objective,
optimizer, and iteration budget. Random shapes and the oval are screening tests
outside the training family; the validated optimization comparison uses the
NACA four-digit parameterization.

Important cavitation caveat: this repository treats cavitation as a secondary
diagnostic and currently derives it from
absolute pressure and vapor pressure. Evaluate pressure-threshold inception risk
over ambient-pressure sweeps with `scripts/evaluate_cavitation_risk.py`. This is
useful for inception risk (`cavitation_margin`, `Cp`), but it is not a
replacement for multiphase cavitating RANS labels such as vapor volume fraction,
cavity length, or shedding dynamics. To train those, add solver outputs such as
`alpha.vapor` to the OpenFOAM extraction and include that field in `--targets`.
