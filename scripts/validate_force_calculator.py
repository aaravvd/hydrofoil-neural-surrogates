#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.hydrodynamic_shape_optimization import Candidate, pressure_force_coefficients


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grid pressure force integration against OpenFOAM forceCoeffs.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "corrected_pilot" / "data" / "processed_grids")
    parser.add_argument("--foam-dir", type=Path, default=ROOT / "corrected_pilot" / "openfoam_cases")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_results" / "force_validation")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.data_dir.glob("case_*_grid.npz")):
        data = np.load(path, allow_pickle=True)
        case_id = path.stem.replace("_grid", "")
        reference = read_force_coefficients(args.foam_dir / case_id)
        if reference is None:
            print(f"[skip] {case_id}: no forceCoeffs.dat")
            continue
        candidate = Candidate(
            candidate_id=case_id,
            family=str(data["naca"]),
            airfoil_x=data["airfoil_x"],
            airfoil_y=data["airfoil_y"],
            AoA=float(data["AoA"]),
            Re=float(data["Re"]),
            rho=float(data["rho"]),
            nu=float(data["nu"]),
            p_inf=float(data["p_inf"]),
            p_vap=float(data["p_vap"]),
            chord=1.0,
        )
        cl_grid, cd_grid = pressure_force_coefficients(
            candidate, {"grid_x": data["grid_x"], "grid_y": data["grid_y"]}, data["p"]
        )
        rows.append(
            {
                "case": case_id,
                "naca": str(data["naca"]),
                "AoA": float(data["AoA"]),
                "Re": float(data["Re"]),
                "Cl_openfoam_total": reference["Cl"],
                "Cl_grid_pressure_only": cl_grid,
                "Cl_abs_error": abs(cl_grid - reference["Cl"]),
                "Cd_openfoam_total": reference["Cd"],
                "Cd_grid_pressure_only": cd_grid,
                "Cd_abs_error": abs(cd_grid - reference["Cd"]),
            }
        )

    if not rows:
        raise SystemExit("No cases with both grids and OpenFOAM force coefficients were found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "force_comparison.csv", rows)
    summary = summarize(rows)
    (args.output_dir / "force_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_parity(args.output_dir / "force_parity.png", rows)
    print(json.dumps(summary, indent=2))


def read_force_coefficients(case_dir: Path) -> dict[str, float] | None:
    paths = sorted((case_dir / "postProcessing" / "forceCoeffs").glob("*/forceCoeffs.dat"))
    if not paths:
        return None
    values = np.loadtxt(paths[-1], comments="#")
    final = values[-1] if values.ndim == 2 else values
    return {"Cm": float(final[1]), "Cd": float(final[2]), "Cl": float(final[3])}


def summarize(rows: list[dict]) -> dict:
    result = {"n_cases": len(rows), "comparison_note": "Grid integral is pressure-only; OpenFOAM reference includes pressure and viscous forces."}
    for name in ["Cl", "Cd"]:
        truth = np.array([row[f"{name}_openfoam_total"] for row in rows])
        pred = np.array([row[f"{name}_grid_pressure_only"] for row in rows])
        result[name] = {
            "mae": float(np.mean(np.abs(pred - truth))),
            "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
            "mean_relative_absolute_error": float(np.mean(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-8))),
            "pearson_r": float(np.corrcoef(pred, truth)[0, 1]) if len(rows) > 1 else float("nan"),
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_parity(path: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for ax, name in zip(axes, ["Cl", "Cd"]):
        truth = np.array([row[f"{name}_openfoam_total"] for row in rows])
        pred = np.array([row[f"{name}_grid_pressure_only"] for row in rows])
        low = min(float(truth.min()), float(pred.min()))
        high = max(float(truth.max()), float(pred.max()))
        ax.scatter(truth, pred, color="#2563eb")
        ax.plot([low, high], [low, high], color="#111827", linestyle="--", linewidth=1)
        ax.set_xlabel(f"OpenFOAM total {name}")
        ax.set_ylabel(f"Grid pressure-only {name}")
        ax.set_title(f"{name} parity")
        ax.grid(alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
