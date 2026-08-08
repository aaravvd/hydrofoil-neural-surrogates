# Hydrodynamic Shape Optimization Screening Report

This is surrogate-assisted design-space screening, not final CFD-validated optimization.
Primary model for headline ranking: `fno`.
Candidates: NACA sweep plus 5 random smooth airfoils across AoA values `-2,0,2,4,6,8` at Re=500000.

Objective: maximize pressure-based `CL / abs(CD)` with penalties for cavitation-margin violations.

## Top Designs: fno

| Rank | Candidate | Family | AoA | CL proxy | CD proxy | CL/abs(CD) | Min cavitation margin | Objective |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | naca_2412_aoa_-2 | NACA 2412 | -2.00 | 0.1253 | -0.0007347 | 170.3 | 9.897e+04 | 170.3 |
| 2 | naca_6409_aoa_-2 | NACA 6409 | -2.00 | 0.07 | 0.0007415 | 94.27 | 9.898e+04 | 94.27 |
| 3 | random_05_aoa_-2 | random_05 | -2.00 | 0.1388 | 0.002642 | 52.53 | 9.894e+04 | 52.53 |
| 4 | naca_4412_aoa_-2 | NACA 4412 | -2.00 | 0.1313 | 0.003037 | 43.22 | 9.894e+04 | 43.22 |
| 5 | naca_2412_aoa_0 | NACA 2412 | 0.00 | 0.1252 | 0.003638 | 34.41 | 9.897e+04 | 34.41 |
| 6 | naca_6409_aoa_0 | NACA 6409 | 0.00 | 0.06997 | 0.003187 | 21.95 | 9.898e+04 | 21.95 |
| 7 | random_05_aoa_0 | random_05 | 0.00 | 0.1387 | 0.007488 | 18.52 | 9.894e+04 | 18.52 |
| 8 | naca_4412_aoa_0 | NACA 4412 | 0.00 | 0.1311 | 0.007618 | 17.21 | 9.894e+04 | 17.21 |
| 9 | naca_2412_aoa_2 | NACA 2412 | 2.00 | 0.1251 | 0.00801 | 15.62 | 9.897e+04 | 15.62 |
| 10 | naca_6409_aoa_2 | NACA 6409 | 2.00 | 0.06987 | 0.00563 | 12.41 | 9.898e+04 | 12.41 |

## Paper Caveat

The force calculation is a pressure-only proxy sampled from gridded surrogate pressure near the airfoil surface. It is useful for ranking and generating candidate designs, but top candidates should be re-run in OpenFOAM before making any hydrodynamic performance claim. Random airfoils are out-of-distribution relative to the current NACA-heavy training set, so they are best presented as a stress test and future validation target.

## Recommended AI4S Framing

Use this section to show downstream scientific workflow value: surrogate field prediction enables rapid screening of hydrofoil design candidates under cavitation-risk constraints, while expensive OpenFOAM is reserved for final validation of the top-ranked designs.