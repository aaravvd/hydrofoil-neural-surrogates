# Hydrofoil ML Paper Results Brief

Generated from `training_runs/` on the deterministic validation split used by the training scripts: 15% validation fraction, seed 7. The validation set contains 58 cases.

## Executive Takeaways

- FNO is the strongest pressure surrogate on the held-out split: RMSE 1.597 Pa, MAE 0.7177 Pa, R2 0.9957.
- DNN, DeepONet, and PINN are close to each other on pressure, with RMSE around 6-7 Pa and R2 around 0.92-0.94.
- Going forward, replace the DNN baseline with CNN-U-Net for the paper. The code now supports `unet`, and the intended model comparison is CNN-U-Net vs PINN vs FNO vs DeepONet.
- FNO also dominates Cp, cavitation margin, turbulence kinetic energy, omega, nut, Rxx, and Ryy, which suggests grid-aware convolution/spectral structure is valuable for these gridded flow fields.
- DeepONet is best on Ux in this run, while FNO is best on Uy. This is a useful nuance for the discussion rather than a simple one-model-wins-everything story.
- The current dataset contains cavitation-risk quantities, but apparently no positive cavitation indicator cells. Treat cavitation as inception-risk/margin prediction, not validated vapor-cavity prediction.
- Rxy is zero or nearly zero in the present data, so Rxy R2 is undefined and it should not be emphasized as evidence of model skill.
- A new HSO screening script ranks NACA and random smooth airfoils using surrogate pressure fields, pressure-force proxy coefficients, and cavitation-margin constraints. Treat this as surrogate-assisted screening; OpenFOAM validation of top candidates is needed before final design claims.
- Runtime benchmarking now parses OpenFOAM `ExecutionTime` logs and measures surrogate inference time for speedup claims.

## Table 1: Pressure and Cavitation-Risk Metrics

| Field | Model | RMSE | MAE | Relative RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| p | fno | 1.597 | 0.7177 | 1.576e-05 | 0.9957 |
| p | dnn | 6.107 | 3.031 | 6.027e-05 | 0.9377 |
| p | deeponet | 6.551 | 3.312 | 6.466e-05 | 0.9283 |
| p | pinn | 6.713 | 3.431 | 6.625e-05 | 0.9247 |
| Cp | fno | 0.006247 | 0.003794 | 0.5534 | 0.9956 |
| Cp | dnn | 0.02296 | 0.01731 | 2.034 | 0.9403 |
| Cp | deeponet | 0.0255 | 0.01973 | 2.259 | 0.9264 |
| Cp | pinn | 0.02623 | 0.02006 | 2.324 | 0.9221 |
| cavitation_margin | fno | 1.597 | 0.7177 | 1.613e-05 | 0.9957 |
| cavitation_margin | dnn | 6.107 | 3.031 | 6.167e-05 | 0.9377 |
| cavitation_margin | deeponet | 6.553 | 3.308 | 6.618e-05 | 0.9282 |
| cavitation_margin | pinn | 6.712 | 3.435 | 6.778e-05 | 0.9247 |
| cavitation_indicator | pinn | 4.393e-07 | 7.301e-09 | 4.393e+05 | n/a |
| cavitation_indicator | dnn | 9.747e-07 | 6.698e-08 | 9.747e+05 | n/a |
| cavitation_indicator | deeponet | 1.245e-06 | 2.437e-08 | 1.245e+06 | n/a |
| cavitation_indicator | fno | 6.468e-05 | 7.160e-06 | 6.468e+07 | n/a |

Note: `cavitation_indicator` predictions are sigmoid probabilities. Since the validation truth is all zero, R2 is not meaningful and classification metrics such as recall cannot be evaluated without positive cavitation cases.

## Table 2: Flow and RANS Field Metrics

| Field | Best Model | RMSE | MAE | R2 | Full Ranking by RMSE |
|---|---:|---:|---:|---:|---|
| Ux | deeponet | 0.01068 | 0.007589 | 0.999 | deeponet (0.01068), dnn (0.01747), pinn (0.02578), fno (0.03092) |
| Uy | fno | 0.002183 | 0.001138 | 0.9929 | fno (0.002183), dnn (0.008262), deeponet (0.008928), pinn (0.009028) |
| nut | fno | 2.340e-06 | 1.167e-06 | 0.996 | fno (2.340e-06), dnn (3.066e-05), pinn (3.136e-05), deeponet (3.193e-05) |
| k | fno | 2.381e-04 | 8.578e-05 | 0.9967 | fno (2.381e-04), deeponet (0.002368), dnn (0.002419), pinn (0.002518) |
| omega | fno | 0.44 | 0.1569 | 0.9962 | fno (0.44), dnn (4.063), deeponet (4.157), pinn (4.295) |
| Rxx | fno | 1.590e-04 | 5.765e-05 | 0.9966 | fno (1.590e-04), deeponet (0.001577), dnn (0.001612), pinn (0.001678) |
| Rxy | dnn | 3.175e-11 | 1.154e-11 | n/a | dnn (3.175e-11), deeponet (2.179e-10), fno (5.725e-10), pinn (4.061e-09) |
| Ryy | fno | 1.587e-04 | 5.709e-05 | 0.9967 | fno (1.587e-04), deeponet (0.001578), dnn (0.001612), pinn (0.001677) |

## Figures to Include

1. Dataset/pipeline schematic: OpenFOAM or analytic smoke generation -> gridding -> normalization -> DNN/PINN/FNO/DeepONet -> evaluation/VV&A.
2. Representative pressure comparison figure: use `figures/model_predictions/case_001_p_predictions.png` or generate 2-3 more validation cases. The figure already includes truth, predictions, and absolute-error maps.
3. Challenging-case pressure comparison: use `case_356` if discussing cavitation margin because it had the lowest margin in the earlier dataset audit, but make clear it still did not cavitate.
4. Bar chart/table of pressure RMSE across models, emphasizing FNO vs pointwise/operator alternatives.
5. Bar chart/table for Cp and cavitation_margin RMSE to connect model performance to hydrofoil cavitation-risk screening.
6. Optional: velocity magnitude or Ux/Uy prediction panel, since DeepONet leads Ux while FNO leads Uy.
7. HSO top-geometry panel: `hso_results/fno_screen/top_geometries.png`.
8. Runtime speedup bar chart from `paper_results/runtime_speedup.csv`.

## Tables to Generate for the Manuscript

- Model architecture table: model, input representation, output fields, inductive bias, trainable width/depth/basis/modes, loss terms.
- Dataset table: number of cases, NACA families, AoA range, Reynolds-number range, grid size, target fields, validation split.
- Primary results table: pressure, Cp, and cavitation_margin RMSE/MAE/R2 for all four models.
- Secondary RANS table: Ux, Uy, k, omega, nut, Rxx, Ryy metrics; omit or footnote Rxy because the target is degenerate.
- Runtime table: OpenFOAM median execution time, model inference time per case, and speedup factor.
- HSO screening table: top NACA/random candidate, AoA, pressure-force proxy CL/CD, minimum cavitation margin, and objective.
- VV&A table: verification checks, validation split protocol, conservation/PINN residual check, visual inspection, cavitation-label limitation.

## JoCSE Framing

For JoCSE, frame this as a student computational-science project comparing data-driven and physics-informed neural surrogates for hydrofoil flow-field learning. The student-paper angle should include project organization, what was learned about CFD/ML/VV&A, implementation challenges, and how the software/artifacts could be reused in a computational science classroom or independent project.

## Claims That Are Supported

- The pipeline can train and compare DNN, PINN, FNO, and DeepONet surrogates on gridded hydrofoil flow data.
- FNO achieved the best pressure, Cp, cavitation_margin, and most RANS-field accuracy on the held-out validation split.
- Cavitation risk can be represented through pressure-derived margin and Cp, and models can predict those scalar fields accurately in this dataset.

## Claims to Avoid

- Do not claim the model predicts real cavitation dynamics, vapor volume fraction, or cavity shedding unless new multiphase CFD labels are generated.
- Do not claim broad generalization to experimental hydrofoil data unless experimental/independent CFD validation is added.
- Do not overinterpret `cavitation_indicator` classification results because there are no positive cavitation cells in the current validation data.

## Files Generated

- `paper_results/field_metrics.csv`: per-model, per-field metrics.
- `paper_results/case_metrics.csv`: per-model, per-case summaries for pressure/Cp/cavitation risk.
- `paper_results/summary.json`: machine-readable summary and model rankings.
- `paper_results/runtime_speedup.csv`: OpenFOAM-vs-surrogate timing table.
- `figures/model_predictions/*.png`: prediction panels for selected cases.
- `hso_results/fno_screen/hso_candidate_scores.csv`: surrogate-assisted HSO candidate rankings.
- `hso_results/fno_screen/hso_report.md`: HSO screening report with top designs and caveats.
