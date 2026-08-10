# Reproducing the Hydrofoil Surrogate Benchmark

## Scope

The paper uses the fixed, disjoint family partition in
`configs/family_split.json`. NACA 0015 and 2412 form the 64-case validation
set used for early stopping. NACA 0018 and 4418 form an untouched 64-case test
set used only after training and checkpoint selection. The other 253 valid
cases train the models. Three CFD cases are removed by the predeclared quality
rule `abs(Cl) <= 5` and `abs(Cd) <= 5`. Every architecture is trained with
seeds 7, 17, and 27 without changing this partition.

The reported target order is:

```text
Ux,Uy,p,nut,k,omega,Cp,cavitation_margin,Cl,Cd
```

## Environment

- Python packages: `requirements-lock.txt`
- CFD: OpenFOAM 9 using `openfoam/openfoam9-graphical-apps`
- Reported hardware: 8-core Apple M2 Mac mini, 16 GB memory
- Reported execution: PyTorch CPU and single-process OpenFOAM in Docker

Create the Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-lock.txt
```

The pinned versions reproduce the environment used for the paper. If a wheel is
not available on another platform, use `requirements-ml.txt` and record the
resolved versions in the experiment log.

## Archived Data and Checkpoints

The corrected CFD dataset and trained checkpoints are archived at
[Zenodo DOI 10.5281/zenodo.21845241](https://doi.org/10.5281/zenodo.21845241).

```text
5bb21a67d2f9fc52ae7556012eae31cc772f9394b0fd0669b7379c96feba71dc  hydrofoil_cfd_dataset_v1.tar.gz
443f162ba38a4d993145fd5c2714823e6c24496129bab237ea31a08f44ac08c8  hydrofoil_model_checkpoints_v1.tar.gz
```

After downloading, verify the files with `shasum -a 256 -c SHA256SUMS.txt`
and extract both archives at the repository root.

## End-to-End Workflow

The stage runner contains every command and exact training hyperparameter:

```bash
./scripts/reproduce_paper.sh data
./scripts/reproduce_paper.sh train
./scripts/reproduce_paper.sh evaluate
./scripts/reproduce_paper.sh optimize
./scripts/reproduce_paper.sh papers
```

Run the complete sequence with:

```bash
./scripts/reproduce_paper.sh all
```

Dataset generation and model training are intentionally separate. This makes it
possible to reproduce the metrics from an archived dataset and checkpoint bundle
without rerunning all OpenFOAM cases.

## Expected Outputs

- `corrected_production/data/processed_grids/`: common-grid RANS cases
- `training_runs/revised/seed_<seed>/<model>/best.pt`: selected checkpoints
- `paper_results/revised/`: per-seed and mean/std final-test metrics
- `hso_results/revised/seed_<seed>/`: all optimizer traces and selected designs
- `hso_results/revised/grid_study/grid_convergence.csv`: key-design grid checks
- `hso_results/corrected/openfoam_baseline/`: matched direct-CFD optimization traces and winners
- `papers/*/main.pdf`: compiled manuscripts

The compact CSV, JSON, and figure outputs are versioned in GitHub. The full CFD
dataset and checkpoint bundle are deposited on Zenodo so every reported table
can be reproduced without storing large binaries in the source repository.

## Claim Boundary

The cavitation task is a secondary pressure-threshold inception-risk diagnostic
derived from single-phase RANS. It is not part of the primary optimization
objective and is not a multiphase prediction of vapor fraction, cavity length,
collapse, or shedding. The continuous optimizer searches the bounded NACA
four-digit parameterization between the discrete training families; CFD endpoint
and grid checks test the selected designs, but do not establish uniform accuracy
throughout that continuous domain.
