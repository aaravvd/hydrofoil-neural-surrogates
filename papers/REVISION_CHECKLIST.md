# Review-Driven Revision Checklist

| Review request | Revision and evidence |
| --- | --- |
| Untouched family-level test set | Frozen in `configs/family_split.json`: NACA 0015/2412 validation; NACA 0018/4418 final test. Training and all evaluators consume the same manifest. |
| Three training seeds | `scripts/reproduce_paper.sh train` runs seeds 7, 17, and 27 for every architecture. `aggregate_multiseed_results.py` reports mean and sample standard deviation. |
| Separate force and pressure errors | Final tables are generated from `field_metrics.csv` and `model_force_summary.json`, including separate `p`, `Cp`, `Cl`, and `Cd` MAE/RMSE values. |
| CFD convergence and grid checks | `grid_convergence_selected_designs.py` runs coarse, medium, and fine meshes and records cells, `Cl`, `Cd`, `L/D`, final residual, and late-iteration force spans. |
| All optimization starts and traces | Every evaluation from NACA 0012, 2412, and 4415 is retained per architecture and seed; `plot_optimization_traces.py` also shows the three direct-CFD histories. |
| Anonymous code/data | `build_anonymous_artifact.py` removes known identity strings and checkpoint paths, includes the 64 test grids plus checkpoints/results, and enforces the 100 MB supplementary limit. |
| Continuous optimizer extrapolation | The methods and discussion distinguish continuous descriptor queries from the twelve discrete CFD families and limit validation claims to CFD-rerun endpoints. |
| Sim2Science format | The paper uses the official unmodified `neurips_2026.sty`, `dblblindworkshop`, `\workshoptitle{Sim2Science}`, anonymous authors, five content pages, and the required checklist. |
| Cavitation claim | Cavitation is demoted to a secondary pressure-threshold diagnostic; the primary HSO objective is explicitly `Cl/abs(Cd)`. |
| Capacity and hyperparameter fairness | The methods state what is standardized, why unequal representations are not artificially parameter-matched, and report parameters, training time, and inference time. |

Before submission, verify that the anonymous full-data destination is populated
and replace `ANONYMOUS_DATA_URL` in the review artifact. Do not put the named
GitHub or Zenodo URL in the double-blind Sim2Science PDF or supplementary ZIP.
