# Hydrodynamic Shape Optimization Screening Report

This is surrogate-assisted design-space screening, not final CFD-validated optimization.
Primary model for headline ranking: `pinn`.
Candidates: NACA sweep plus 5 random smooth airfoils across AoA values `-2,0,2,4,6,8` at Re=500000.

Objective: maximize surrogate `CL / abs(CD)` with penalties for cavitation-margin violations.

## Top Designs: unet

| Rank | Candidate | Family | AoA | CL surrogate | CD surrogate | CL/abs(CD) | Min cavitation margin | Objective |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | naca_6409_aoa_2 | NACA 6409 | 2.00 | 0.6295 | 0.02273 | 27.7 | 9.892e+04 | 27.7 |
| 2 | naca_6409_aoa_0 | NACA 6409 | 0.00 | 0.4779 | 0.01871 | 25.54 | 9.895e+04 | 25.54 |
| 3 | random_05_aoa_4 | random_05 | 4.00 | 0.7099 | 0.02855 | 24.86 | 9.89e+04 | 24.86 |
| 4 | naca_6409_aoa_4 | NACA 6409 | 4.00 | 0.8186 | 0.033 | 24.8 | 9.888e+04 | 24.8 |
| 5 | naca_4412_aoa_2 | NACA 4412 | 2.00 | 0.636 | 0.0258 | 24.65 | 9.891e+04 | 24.65 |
| 6 | naca_4412_aoa_4 | NACA 4412 | 4.00 | 0.8017 | 0.03281 | 24.44 | 9.888e+04 | 24.44 |
| 7 | random_05_aoa_6 | random_05 | 6.00 | 0.8888 | 0.03733 | 23.81 | 9.886e+04 | 23.81 |
| 8 | random_03_aoa_4 | random_03 | 4.00 | 0.5556 | 0.02348 | 23.66 | 9.891e+04 | 23.66 |
| 9 | random_02_aoa_2 | random_02 | 2.00 | 0.5852 | 0.02508 | 23.33 | 9.889e+04 | 23.33 |
| 10 | random_05_aoa_2 | random_05 | 2.00 | 0.5493 | 0.02354 | 23.33 | 9.892e+04 | 23.33 |

## Top Designs: pinn

| Rank | Candidate | Family | AoA | CL surrogate | CD surrogate | CL/abs(CD) | Min cavitation margin | Objective |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | naca_6409_aoa_2 | NACA 6409 | 2.00 | 0.8574 | 0.02179 | 39.35 | 9.89e+04 | 39.35 |
| 2 | naca_8412_aoa_2 | NACA 8412 | 2.00 | 1.066 | 0.02854 | 37.37 | 9.887e+04 | 37.37 |
| 3 | naca_6409_aoa_0 | NACA 6409 | 0.00 | 0.6755 | 0.01874 | 36.03 | 9.892e+04 | 36.03 |
| 4 | naca_6409_aoa_4 | NACA 6409 | 4.00 | 1.021 | 0.02922 | 34.93 | 9.889e+04 | 34.93 |
| 5 | naca_8412_aoa_0 | NACA 8412 | 0.00 | 0.8959 | 0.0259 | 34.59 | 9.889e+04 | 34.59 |
| 6 | naca_8412_aoa_4 | NACA 8412 | 4.00 | 1.207 | 0.03622 | 33.32 | 9.884e+04 | 33.32 |
| 7 | random_02_aoa_2 | random_02 | 2.00 | 0.7375 | 0.02492 | 29.59 | 9.89e+04 | 29.59 |
| 8 | naca_4412_aoa_2 | NACA 4412 | 2.00 | 0.611 | 0.02068 | 29.54 | 9.892e+04 | 29.54 |
| 9 | naca_6409_aoa_6 | NACA 6409 | 6.00 | 1.181 | 0.04038 | 29.25 | 9.886e+04 | 29.25 |
| 10 | naca_4412_aoa_4 | NACA 4412 | 4.00 | 0.7942 | 0.02734 | 29.05 | 9.889e+04 | 29.05 |

## Top Designs: fno

| Rank | Candidate | Family | AoA | CL surrogate | CD surrogate | CL/abs(CD) | Min cavitation margin | Objective |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | naca_8412_aoa_4 | NACA 8412 | 4.00 | 1.042 | 0.04508 | 23.1 | 9.885e+04 | 23.1 |
| 2 | random_02_aoa_4 | random_02 | 4.00 | 0.9857 | 0.04324 | 22.8 | 9.885e+04 | 22.8 |
| 3 | naca_8412_aoa_2 | NACA 8412 | 2.00 | 0.8526 | 0.03756 | 22.7 | 9.888e+04 | 22.7 |
| 4 | naca_8412_aoa_6 | NACA 8412 | 6.00 | 1.223 | 0.0543 | 22.53 | 9.88e+04 | 22.53 |
| 5 | random_02_aoa_6 | random_02 | 6.00 | 1.163 | 0.05172 | 22.49 | 9.88e+04 | 22.49 |
| 6 | random_02_aoa_2 | random_02 | 2.00 | 0.8085 | 0.0362 | 22.33 | 9.888e+04 | 22.33 |
| 7 | random_02_aoa_8 | random_02 | 8.00 | 1.317 | 0.06105 | 21.57 | 9.87e+04 | 21.57 |
| 8 | random_05_aoa_6 | random_05 | 6.00 | 0.835 | 0.03925 | 21.28 | 9.888e+04 | 21.28 |
| 9 | naca_8412_aoa_8 | NACA 8412 | 8.00 | 1.353 | 0.06388 | 21.17 | 9.859e+04 | 21.17 |
| 10 | naca_8412_aoa_0 | NACA 8412 | 0.00 | 0.6696 | 0.03174 | 21.09 | 9.891e+04 | 21.09 |

## Top Designs: deeponet

| Rank | Candidate | Family | AoA | CL surrogate | CD surrogate | CL/abs(CD) | Min cavitation margin | Objective |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | naca_6409_aoa_4 | NACA 6409 | 4.00 | 0.8606 | 0.025 | 34.42 | 9.888e+04 | 34.42 |
| 2 | naca_8412_aoa_4 | NACA 8412 | 4.00 | 0.9462 | 0.02789 | 33.92 | 9.886e+04 | 33.92 |
| 3 | naca_8412_aoa_6 | NACA 8412 | 6.00 | 1.148 | 0.03583 | 32.03 | 9.88e+04 | 32.03 |
| 4 | naca_6409_aoa_2 | NACA 6409 | 2.00 | 0.6523 | 0.02044 | 31.91 | 9.892e+04 | 31.91 |
| 5 | naca_4412_aoa_4 | NACA 4412 | 4.00 | 0.803 | 0.02523 | 31.82 | 9.89e+04 | 31.82 |
| 6 | random_02_aoa_2 | random_02 | 2.00 | 0.7362 | 0.02314 | 31.82 | 9.89e+04 | 31.82 |
| 7 | naca_4412_aoa_2 | NACA 4412 | 2.00 | 0.6041 | 0.01909 | 31.64 | 9.893e+04 | 31.64 |
| 8 | random_02_aoa_4 | random_02 | 4.00 | 0.9318 | 0.02994 | 31.12 | 9.887e+04 | 31.12 |
| 9 | naca_8412_aoa_2 | NACA 8412 | 2.00 | 0.7372 | 0.02399 | 30.72 | 9.89e+04 | 30.72 |
| 10 | naca_6409_aoa_6 | NACA 6409 | 6.00 | 1.06 | 0.03456 | 30.67 | 9.884e+04 | 30.67 |

## Paper Caveat

When checkpoints include direct OpenFOAM `Cl` and `Cd` targets, ranking uses those outputs; older checkpoints fall back to a pressure-only grid integral. In either case, top candidates should be re-run in OpenFOAM before making any hydrodynamic performance claim. Random airfoils are out-of-distribution relative to the current NACA-heavy training set, so they are best presented as a stress test and future validation target.

## Recommended AI4S Framing

Use this section to show downstream scientific workflow value: surrogate field prediction enables rapid screening of hydrofoil design candidates under cavitation-risk constraints, while expensive OpenFOAM is reserved for final validation of the top-ranked designs.