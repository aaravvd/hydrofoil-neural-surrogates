# Claim and Evidence Map

This file is an author-side audit aid and is not part of the submission.

| Manuscript claim | Evidence artifact | Qualification |
|---|---|---|
| 381 usable cases and 64 held-out cases | `training_runs/corrected/*/metrics.json` | Three cases with nonphysical force coefficients above 5 were excluded. |
| NACA 0015 and 2412 were fully held out | Deterministic `split_paths_by_naca(..., 0.15, 7)` | This tests unseen NACA families, not arbitrary geometry generalization. |
| PINN-style model has pressure R2 0.966 | `paper_results/corrected/field_metrics.csv` | Call it a supervised physics-regularized point network, not a classical boundary-only PINN. |
| DeepONet has best Cl/Cd R2 | `paper_results/corrected/field_metrics.csv` | Values are 0.9964 and 0.9883 on the held-out geometry split. |
| Speedups are 917x--1801x | `paper_results/corrected/runtime_speedup.csv` | Inference timing predicts the pressure field; OpenFOAM time is parsed from solver logs on the same local system. |
| Cavitation-risk F1 is 0.771--0.921 | `paper_results/corrected/cavitation_risk/cavitation_risk_summary.json` | Pressure-threshold inception risk under ambient-pressure sweeps, not multiphase cavity dynamics. |
| Optimized winners improve CFD L/D by 2.3%--79.7% | `hso_results/corrected/optimized/openfoam_validation.csv` | Each model's selected winner was rerun in OpenFOAM; comparisons use that start foil as baseline. |
| Random and oval profiles demonstrate OOD behavior | `hso_results/corrected/screening/hso_candidate_scores.csv` | Stress tests only; do not present as validated free-form optimization. |

## Required Author Checks

- Replace author and affiliation placeholders.
- Confirm whether code and data can be anonymized and released.
- State the exact CPU/GPU and OpenFOAM version used for runtime experiments.
- Confirm travel and registration feasibility before final submission.
- Run an external similarity and citation check before submission.
