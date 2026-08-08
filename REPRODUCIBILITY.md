# Reproducing the Hydrofoil Surrogate Benchmark

## Scope

The paper reports a deterministic NACA-family validation split with seed 7.
NACA 0015 and NACA 2412 form the 64-case validation set. The other 317 valid
cases are used for training. Three CFD cases are removed by the predeclared
quality rule `abs(Cl) <= 5` and `abs(Cd) <= 5`.

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
- `training_runs/corrected/<model>/best.pt`: selected checkpoints
- `paper_results/corrected/`: field, force, cavitation, runtime, and loss tables
- `hso_results/corrected/optimized/`: optimizer traces and selected designs
- `hso_results/corrected/optimized/openfoam_validation.csv`: CFD reruns
- `hso_results/corrected/openfoam_baseline/`: matched direct-CFD optimization traces and winners
- `papers/*/main.pdf`: compiled manuscripts

The compact CSV, JSON, and figure outputs are versioned in GitHub. The full CFD
dataset and checkpoint bundle are deposited on Zenodo so every reported table
can be reproduced without storing large binaries in the source repository.

## Claim Boundary

The cavitation task is pressure-threshold inception-risk screening derived from
single-phase RANS. It is not a multiphase prediction of vapor fraction, cavity
length, collapse, or shedding. The random profiles and oval are tests outside
the training geometry family; the paper's validated optimization claims concern
the NACA parameterization and the designs rerun in OpenFOAM.
