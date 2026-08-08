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
- `papers/*/main.pdf`: compiled manuscripts

The compact CSV, JSON, and figure outputs should be versioned in GitHub. The CFD
dataset and checkpoint bundle should be deposited in an archival service such as
Zenodo and linked from the GitHub release. Add SHA-256 checksums and replace the
repository URL and release version to `CITATION.cff` before the public release.

## Claim Boundary

The cavitation task is pressure-threshold inception-risk screening derived from
single-phase RANS. It is not a multiphase prediction of vapor fraction, cavity
length, collapse, or shedding. The random profiles and oval are tests outside
the training geometry family; the paper's validated optimization claims concern
the NACA parameterization and the designs rerun in OpenFOAM.
