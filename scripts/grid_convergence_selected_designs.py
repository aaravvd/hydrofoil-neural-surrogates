#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
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

GRIDS = {
    "coarse": (0.025, 1.5),
    "medium": (0.015, 1.0),
    "fine": (0.010, 0.75),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-level CFD grid and iterative-convergence check for selected designs.")
    parser.add_argument("--optimization-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hso_results" / "revised" / "grid_study")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--Re", type=float, default=500000.0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--include-models", default="unet,pinn,fno,deeponet")
    args = parser.parse_args()

    requested = {item.strip() for item in args.include_models.split(",") if item.strip()}
    winners = select_winners(args.optimization_summary, requested)
    jobs = [(model, row, level, sizes) for model, row in winners.items() for level, sizes in GRIDS.items()]
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_design, args, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[ok] {row['model']} {row['grid_level']}: {row['cells']} cells")
    rows.sort(key=lambda row: (row["model"], list(GRIDS).index(row["grid_level"])))
    add_relative_changes(rows)
    write_csv(args.output_dir / "grid_convergence.csv", rows)


def select_winners(path: Path, requested: set[str]) -> dict[str, dict]:
    winners: dict[str, dict] = {}
    for row in csv.DictReader(path.open()):
        model = row["model"]
        if model not in requested:
            continue
        if model not in winners or float(row["best_objective"]) > float(winners[model]["best_objective"]):
            winners[model] = row
    if winners.keys() != requested:
        raise ValueError(f"Missing optimization winners for {sorted(requested - winners.keys())}")
    return winners


def run_design(args, model: str, row: dict, level: str, sizes: tuple[float, float]) -> dict:
    lc_foil, lc_far = sizes
    case = Case(
        id=f"{model}_{level}", naca="custom", AoA=float(row["best_AoA"]), Re=args.Re,
        rho=997.0, nu=1e-6, p_inf=101325.0, p_vap=2300.0, chord=1.0,
    )
    coords = coordinates_from_parameters(float(row["best_m"]), float(row["best_p"]), float(row["best_t"]), n=121)
    case_dir = args.output_dir / "openfoam_cases" / case.id
    write_case(
        case, case_dir, airfoil_coords=coords, mesh_lc_foil=lc_foil,
        mesh_lc_far=lc_far, end_time=args.iterations,
    )
    run_openfoam_case(case_dir, ROOT, use_docker=True)
    force = read_force_coefficients(case_dir)
    residual, cl_span, cd_span = convergence_metrics(case_dir)
    return {
        "model": model, "grid_level": level, "iterations": args.iterations,
        "lc_foil": lc_foil, "lc_far": lc_far,
        "cells": mesh_cells(case_dir), "Cl": force["Cl"], "Cd": force["Cd"],
        "L_over_D": force["Cl"] / max(abs(force["Cd"]), 1e-12),
        "max_initial_residual_final_iteration": residual,
        "Cl_span_final_50_iterations": cl_span, "Cd_span_final_50_iterations": cd_span,
    }


def mesh_cells(case_dir: Path) -> int:
    text = (case_dir / "log.checkMesh").read_text(errors="replace")
    matches = re.findall(r"^\s*cells:\s*(\d+)", text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else -1


def convergence_metrics(case_dir: Path) -> tuple[float, float, float]:
    text = (case_dir / "log.simpleFoam").read_text(errors="replace")
    times = list(re.finditer(r"^Time = (\d+)\s*$", text, flags=re.MULTILINE))
    block = text[times[-1].start():] if times else text
    residuals = [float(value) for value in re.findall(r"Initial residual = ([0-9.eE+\-]+)", block)]
    force_paths = sorted(case_dir.glob("postProcessing/forceCoeffs/*/forceCoeffs.dat"))
    values = np.loadtxt(force_paths[-1], comments="#", ndmin=2)
    tail = values[values[:, 0] >= values[-1, 0] - 50]
    return max(residuals) if residuals else np.nan, float(np.ptp(tail[:, 3])), float(np.ptp(tail[:, 2]))


def add_relative_changes(rows: list[dict]) -> None:
    by_model = {model: {row["grid_level"]: row for row in rows if row["model"] == model} for model in {row["model"] for row in rows}}
    for levels in by_model.values():
        fine = levels["fine"]
        for row in levels.values():
            for metric in ("Cl", "Cd", "L_over_D"):
                row[f"{metric}_relative_error_vs_fine"] = abs(row[metric] - fine[metric]) / max(abs(fine[metric]), 1e-12)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
