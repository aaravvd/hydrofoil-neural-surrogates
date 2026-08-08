#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrofoil_pipeline.cases import Case
from hydrofoil_pipeline.naca import coordinates_from_parameters
from hydrofoil_pipeline.openfoam import run_openfoam_case, write_case
from scripts.validate_force_calculator import read_force_coefficients


def main() -> None:
    parser = argparse.ArgumentParser(description="Revalidate each model's optimized winner with OpenFOAM.")
    parser.add_argument("--optimization-summary", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "hso_results" / "corrected" / "openfoam_validation")
    parser.add_argument("--output", type=Path, default=ROOT / "hso_results" / "corrected" / "optimized" / "openfoam_validation.csv")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--Re", type=float, default=500000.0)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.optimization_summary.open()))
    winners = {}
    for row in rows:
        if row["model"] not in winners or float(row["best_objective"]) > float(winners[row["model"]]["best_objective"]):
            winners[row["model"]] = row
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    def run(model, row):
        case = Case(
            id=f"winner_{model}",
            naca="custom",
            AoA=float(row["best_AoA"]),
            Re=args.Re,
            rho=997.0,
            nu=1e-6,
            p_inf=101325.0,
            p_vap=2300.0,
            chord=1.0,
        )
        coords = coordinates_from_parameters(float(row["best_m"]), float(row["best_p"]), float(row["best_t"]), n=121)
        case_dir = args.artifact_root / "openfoam_cases" / case.id
        write_case(case, case_dir, airfoil_coords=coords)
        run_openfoam_case(case_dir, ROOT, use_docker=True)
        force = read_force_coefficients(case_dir)
        return {
            "model": model,
            "start": row["start"],
            "m": row["best_m"],
            "p": row["best_p"],
            "t": row["best_t"],
            "AoA": row["best_AoA"],
            "predicted_Cl": row["best_CL"],
            "predicted_Cd": row["best_CD"],
            "predicted_L_over_D": row["best_L_over_D"],
            "openfoam_Cl": force["Cl"],
            "openfoam_Cd": force["Cd"],
            "openfoam_L_over_D": force["Cl"] / max(abs(force["Cd"]), 1e-9),
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run, model, row): model for model, row in winners.items()}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[ok] validated {result['model']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["model"]))
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
