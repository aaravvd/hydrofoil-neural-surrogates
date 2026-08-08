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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluate_model_results import load_checkpoints, predict_case, require_torch
from scripts.hydrodynamic_shape_optimization import Candidate, pressure_force_coefficients
from scripts.validate_force_calculator import read_force_coefficients
from models.datasets import filter_force_outliers, split_paths, split_paths_by_naca


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare surrogate-derived forces with CFD-grid and OpenFOAM references.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--foam-dir", type=Path, required=True)
    parser.add_argument("--models", default="unet,pinn,fno,deeponet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_results" / "model_forces")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-abs-force-coefficient", type=float, default=5.0)
    parser.add_argument("--split", choices=["validation", "train", "all"], default="validation")
    parser.add_argument("--split-strategy", choices=["random", "naca"], default="naca")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    bundles = load_checkpoints(torch, args.run_dir, selected, device)
    rows = []
    for model_name, bundle in bundles.items():
        if "p" not in bundle["norm"].target_fields:
            print(f"[skip] {model_name}: checkpoint does not predict pressure")
            continue
        p_index = bundle["norm"].target_fields.index("p")
        cl_index = bundle["norm"].target_fields.index("Cl") if "Cl" in bundle["norm"].target_fields else None
        cd_index = bundle["norm"].target_fields.index("Cd") if "Cd" in bundle["norm"].target_fields else None
        paths, _ = filter_force_outliers(sorted(args.data_dir.glob("case_*_grid.npz")), args.max_abs_force_coefficient)
        splitter = split_paths_by_naca if args.split_strategy == "naca" else split_paths
        train_paths, val_paths = splitter(paths, args.val_fraction, args.seed)
        paths = val_paths if args.split == "validation" else train_paths if args.split == "train" else paths
        for path in paths:
            truth, pred, mask, meta = predict_case(torch, bundle, path, args.batch_size, device)
            data = np.load(path, allow_pickle=True)
            candidate = candidate_from_grid(path, data)
            grid = {"grid_x": data["grid_x"], "grid_y": data["grid_y"]}
            truth_pressure = np.where(mask, truth[p_index], np.nan)
            model_pressure = np.where(mask, pred[p_index], np.nan)
            cl_grid, cd_grid = pressure_force_coefficients(candidate, grid, truth_pressure)
            cl_integrated, cd_integrated = pressure_force_coefficients(candidate, grid, model_pressure)
            cl_model = float(np.nanmedian(pred[cl_index][mask])) if cl_index is not None else cl_integrated
            cd_model = float(np.nanmedian(pred[cd_index][mask])) if cd_index is not None else cd_integrated
            openfoam = read_force_coefficients(args.foam_dir / meta["case"])
            rows.append(
                {
                    "model": model_name,
                    **meta,
                    "Cl_model_pressure": cl_model,
                    "Cl_model_integrated_pressure": cl_integrated,
                    "Cl_grid_pressure": cl_grid,
                    "Cl_openfoam_total": openfoam["Cl"] if openfoam else "",
                    "Cd_model_pressure": cd_model,
                    "Cd_model_integrated_pressure": cd_integrated,
                    "Cd_grid_pressure": cd_grid,
                    "Cd_openfoam_total": openfoam["Cd"] if openfoam else "",
                    "L_over_D_model_pressure": cl_model / max(abs(cd_model), 1e-6),
                    "L_over_D_openfoam_total": openfoam["Cl"] / max(abs(openfoam["Cd"]), 1e-6) if openfoam else "",
                }
            )

    if not rows:
        raise SystemExit("No model force rows generated.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "model_force_cases.csv", rows)
    summary = summarize(rows)
    (args.output_dir / "model_force_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_force_parity(args.output_dir / "model_force_parity.png", rows)
    print(json.dumps(summary, indent=2))


def candidate_from_grid(path: Path, data) -> Candidate:
    return Candidate(
        candidate_id=path.stem.replace("_grid", ""),
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


def summarize(rows: list[dict]) -> dict:
    result = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        metrics = {"n_cases": len(subset)}
        for coefficient in ["Cl", "Cd"]:
            pred = np.array([float(row[f"{coefficient}_model_pressure"]) for row in subset])
            grid = np.array([float(row[f"{coefficient}_grid_pressure"]) for row in subset])
            metrics[f"{coefficient}_vs_grid_pressure"] = errors(pred, grid)
            valid = [row for row in subset if row[f"{coefficient}_openfoam_total"] != ""]
            if valid:
                pred_valid = np.array([float(row[f"{coefficient}_model_pressure"]) for row in valid])
                foam = np.array([float(row[f"{coefficient}_openfoam_total"]) for row in valid])
                metrics[f"{coefficient}_vs_openfoam_total"] = errors(pred_valid, foam)
        result[model] = metrics
    return result


def errors(pred: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "mae": float(np.mean(np.abs(pred - truth))),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "pearson_r": float(np.corrcoef(pred, truth)[0, 1]) if pred.size > 1 else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_force_parity(path: Path, rows: list[dict]) -> None:
    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(2, len(models), figsize=(3.2 * len(models), 6), constrained_layout=True, squeeze=False)
    for column, model in enumerate(models):
        subset = [row for row in rows if row["model"] == model and row["Cl_openfoam_total"] != ""]
        for row_index, coefficient in enumerate(["Cl", "Cd"]):
            truth = np.array([float(row[f"{coefficient}_openfoam_total"]) for row in subset])
            pred = np.array([float(row[f"{coefficient}_model_pressure"]) for row in subset])
            ax = axes[row_index, column]
            ax.scatter(truth, pred, s=8, alpha=0.45, color="#2563eb")
            low, high = min(truth.min(), pred.min()), max(truth.max(), pred.max())
            ax.plot([low, high], [low, high], "--", color="#111827", linewidth=1)
            ax.set_title(f"{model}: {coefficient}")
            ax.set_xlabel("OpenFOAM")
            ax.set_ylabel("Surrogate")
            ax.grid(alpha=0.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
