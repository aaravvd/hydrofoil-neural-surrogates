# Corrected Hydrofoil Benchmark Results

## Evidence Status

| Artifact | Status |
|---|---|
| Corrected dataset audit | ready |
| Held-out field metrics | ready |
| Runtime benchmark | ready |
| Model force validation | ready |
| Cavitation-risk evaluation | ready |
| Continuous HSO results | ready |

## Held-Out Field Accuracy

| Model | Field | RMSE | MAE | R2 |
|---|---|---:|---:|---:|
| unet | Ux | 0.3501 | 0.2982 | -0.001181 |
| unet | Uy | 0.03921 | 0.02615 | 0.5491 |
| unet | p | 28.52 | 13.67 | 0.3917 |
| unet | Cp | 0.02941 | 0.01994 | 0.957 |
| unet | Cl | 0.1094 | 0.08677 | 0.9469 |
| unet | Cd | 0.004848 | 0.00396 | 0.9227 |
| pinn | Ux | 0.02284 | 0.01641 | 0.9957 |
| pinn | Uy | 0.01079 | 0.005645 | 0.9659 |
| pinn | p | 6.784 | 3.009 | 0.9656 |
| pinn | Cp | 0.02676 | 0.01295 | 0.9644 |
| pinn | Cl | 0.04005 | 0.03084 | 0.9929 |
| pinn | Cd | 0.00239 | 0.001881 | 0.9812 |
| fno | Ux | 0.3267 | 0.2731 | 0.1284 |
| fno | Uy | 0.02883 | 0.01896 | 0.7562 |
| fno | p | 22.73 | 11.08 | 0.6139 |
| fno | Cp | 0.03846 | 0.02257 | 0.9264 |
| fno | Cl | 0.1466 | 0.1197 | 0.9046 |
| fno | Cd | 0.009766 | 0.007413 | 0.6864 |
| deeponet | Ux | 0.01958 | 0.01319 | 0.9969 |
| deeponet | Uy | 0.01038 | 0.004907 | 0.9684 |
| deeponet | p | 8.546 | 2.912 | 0.9454 |
| deeponet | Cp | 0.03308 | 0.01484 | 0.9456 |
| deeponet | Cl | 0.02859 | 0.02211 | 0.9964 |
| deeponet | Cd | 0.00189 | 0.001542 | 0.9883 |

## Training Cost

| Model | Parameters | Best epoch | Training time (s) | Best validation objective |
|---|---:|---:|---:|---:|
| unet | 485178 | 120 | 867.9 | 3.763 |
| pinn | 9930 | 30 | 104 | 1.426 |
| fno | 595914 | 54 | 547.1 | 3.292 |
| deeponet | 100746 | 6 | 53.77 | 1.505 |

## Runtime

| Model | Inference (s/case) | OpenFOAM (s/case) | Speedup |
|---|---:|---:|---:|
| unet | 0.0108 | 12.4 | 1148x |
| pinn | 0.006883 | 12.4 | 1801x |
| fno | 0.01206 | 12.4 | 1028x |
| deeponet | 0.01352 | 12.4 | 917.5x |

## Force Accuracy

| Model | Cl RMSE vs OpenFOAM | Cd RMSE vs OpenFOAM |
|---|---:|---:|
| deeponet | 0.02681 | 0.001838 |
| fno | 0.1383 | 0.009568 |
| pinn | 0.02451 | 0.002043 |
| unet | 0.09912 | 0.004534 |

## Incipient Cavitation Risk

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| deeponet | 0.9069 | 0.9186 | 0.9127 |
| fno | 0.6819 | 0.8868 | 0.771 |
| pinn | 0.9267 | 0.9162 | 0.9214 |
| unet | 0.7962 | 0.9576 | 0.8695 |

This is pressure-threshold onset risk from single-phase RANS Cp, not multiphase cavity evolution.

## Shape Optimization

| Model | Start | m | p | t | AoA | Best L/D | Min margin (Pa) |
|---|---|---:|---:|---:|---:|---:|---:|
| unet | 0012 | 2.016e-07 | 0.4024 | 0.1193 | 4.078 | 17.82 | 9.891e+04 |
| unet | 2412 | 0.03249 | 0.5498 | 0.06 | 3.058 | 30.17 | 9.891e+04 |
| unet | 4415 | 0.05377 | 0.7496 | 0.06315 | 3.803 | 28.08 | 9.893e+04 |
| pinn | 0012 | 0.00509 | 0.203 | 0.08768 | 5.439 | 22.35 | 9.895e+04 |
| pinn | 2412 | 0.07511 | 0.3559 | 0.06 | 1.156 | 43.55 | 9.891e+04 |
| pinn | 4415 | 0.06667 | 0.3877 | 0.06016 | 1.235 | 43.62 | 9.893e+04 |
| fno | 0012 | 0.0007036 | 0.1777 | 0.1418 | 4.833 | 16.18 | 9.882e+04 |
| fno | 2412 | 0.01842 | 0.3805 | 0.1225 | 4.452 | 18.44 | 9.887e+04 |
| fno | 4415 | 0.03912 | 0.372 | 0.1682 | 4.233 | 21.11 | 9.885e+04 |
| deeponet | 0012 | 0.01762 | 0.2922 | 0.06 | 0.2261 | 33.63 | 9.896e+04 |
| deeponet | 2412 | 0.03349 | 0.3279 | 0.0604 | 1.603 | 39.99 | 9.894e+04 |
| deeponet | 4415 | 0.05911 | 0.3292 | 0.08462 | 2.968 | 36.17 | 9.888e+04 |

## OpenFOAM-Revalidated Optimization

| Model | Baseline | Predicted L/D | OpenFOAM L/D | Improvement vs baseline |
|---|---|---:|---:|---:|
| deeponet | NACA 2412 (25.89) | 39.99 | 40.86 | 57.82% |
| fno | NACA 4415 (25.57) | 21.11 | 26.15 | 2.259% |
| pinn | NACA 4415 (25.57) | 43.62 | 45.96 | 79.73% |
| unet | NACA 2412 (25.89) | 30.17 | 38.45 | 48.49% |

## Paper Figures and Tables

1. Dataset/design-space table: NACA families, Reynolds numbers, AoA values, grid size, train/validation split.
2. Training and validation loss curves for all four models.
3. Pressure and Cp truth/prediction/error maps on interpolation and held-out-geometry cases.
4. Field accuracy table with RMSE, MAE, R2, parameter count, and training time.
5. OpenFOAM versus surrogate runtime and speedup table.
6. Cl/Cd parity plots against direct OpenFOAM forceCoeffs.
7. Cavitation-risk precision/recall/F1 and representative margin maps.
8. HSO convergence curves, optimized profiles, cross-model ranking agreement, and OpenFOAM revalidation of winners.
9. Oval and random-shape out-of-distribution stress-test table, reported separately from in-distribution optimization.

