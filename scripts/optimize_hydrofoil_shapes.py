#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrofoil_pipeline.naca import coordinates_from_parameters, parse_naca4
from scripts.hydrodynamic_shape_optimization import (
    MODEL_ORDER,
    load_checkpoints,
    make_candidate,
    make_grid,
    predict_candidate,
    require_torch,
    score_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-by-model continuous NACA shape optimization using scipy.minimize.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hso_results" / "optimized")
    parser.add_argument("--models", default="unet,pinn,fno,deeponet")
    parser.add_argument("--starts", default="0012,2412,4415")
    parser.add_argument("--Re", type=float, default=500000.0)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--grid-nx", type=int, default=128)
    parser.add_argument("--grid-ny", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--cavitation-margin-threshold", type=float, default=5000.0)
    parser.add_argument(
        "--cavitation-mode",
        choices=["diagnostic", "penalty"],
        default="diagnostic",
        help="Report cavitation margin only, or include its penalty in the optimization objective.",
    )
    parser.add_argument("--rho", type=float, default=997.0)
    parser.add_argument("--nu", type=float, default=1e-6)
    parser.add_argument("--p-inf", type=float, default=101325.0)
    parser.add_argument("--p-vap", type=float, default=2300.0)
    parser.add_argument("--chord", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    requested = MODEL_ORDER if args.models == "auto" else [item.strip() for item in args.models.split(",") if item.strip()]
    bundles = load_checkpoints(torch, args.run_dir, requested, device)
    grid = make_grid(args.grid_nx, args.grid_ny)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    results = []

    for model_name, bundle in bundles.items():
        for start_code in [item.strip() for item in args.starts.split(",") if item.strip()]:
            foil = parse_naca4(start_code)
            x0 = np.array([foil.m, max(foil.p, 0.4), foil.t, 4.0])
            evaluation = 0

            def objective(values):
                nonlocal evaluation
                evaluation += 1
                m, p, t, aoa = values
                ax, ay = coordinates_from_parameters(m, p, t, n=241, chord=args.chord)
                candidate = make_candidate(f"continuous_{evaluation}", "continuous_naca", ax, ay, aoa, args)
                prediction = predict_candidate(torch, bundle, candidate, grid, args.batch_size, device)
                row = score_candidate(
                    model_name,
                    candidate,
                    prediction,
                    grid,
                    args.cavitation_margin_threshold,
                    apply_cavitation_penalty=args.cavitation_mode == "penalty",
                )
                traces.append(
                    {"model": model_name, "start": start_code, "evaluation": evaluation, "m": m, "p": p, "t": t, "AoA": aoa, **row}
                )
                return -float(row["objective"])

            result = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                bounds=[(0.0, 0.09), (0.1, 0.9), (0.06, 0.20), (-4.0, 10.0)],
                options={"maxiter": args.maxiter, "xatol": 5e-4, "fatol": 1e-3},
            )
            best = min((row for row in traces if row["model"] == model_name and row["start"] == start_code), key=lambda row: -row["objective"])
            results.append(
                {
                    "model": model_name,
                    "start": start_code,
                    "success": bool(result.success),
                    "message": str(result.message),
                    "evaluations": int(result.nfev),
                    "best_m": best["m"],
                    "best_p": best["p"],
                    "best_t": best["t"],
                    "best_AoA": best["AoA"],
                    "best_CL": best["CL_surrogate"],
                    "best_CD": best["CD_surrogate"],
                    "best_L_over_D": best["CL_over_abs_CD"],
                    "best_min_cavitation_margin": best["min_cavitation_margin"],
                    "best_objective": best["objective"],
                    "cavitation_mode": args.cavitation_mode,
                }
            )

    write_csv(args.output_dir / "optimization_trace.csv", traces)
    write_csv(args.output_dir / "optimization_summary.csv", results)
    (args.output_dir / "optimization_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"[ok] wrote {args.output_dir / 'optimization_summary.csv'}")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
