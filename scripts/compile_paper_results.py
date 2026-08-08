#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile paper-ready tables and an evidence/status report.")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "paper_results" / "corrected")
    parser.add_argument("--hso-dir", type=Path, default=ROOT / "hso_results" / "corrected")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs" / "corrected")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "corrected_production" / "data" / "processed_grids")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_results" / "corrected_results_report.md")
    args = parser.parse_args()

    field_metrics = read_csv(args.results_dir / "field_metrics.csv")
    runtime = read_csv(args.results_dir / "runtime_speedup.csv")
    force_summary = read_json(args.results_dir / "model_forces" / "model_force_summary.json")
    cavitation = read_json(args.results_dir / "cavitation_risk" / "cavitation_risk_summary.json")
    optimization = read_csv(args.hso_dir / "optimization_summary.csv")
    hso_validation = read_csv(args.hso_dir / "openfoam_validation.csv")
    training = {model: read_json(args.run_dir / model / "metrics.json") for model in ["unet", "pinn", "fno", "deeponet"]}
    audit = read_json(args.results_dir / "dataset_audit.json")

    lines = ["# Corrected Hydrofoil Benchmark Results", ""]
    lines.extend(status_section(audit, field_metrics, runtime, force_summary, cavitation, optimization))
    if field_metrics:
        lines.extend(field_table(field_metrics))
    if any(training.values()):
        lines.extend(training_table(training))
    if runtime:
        lines.extend(runtime_table(runtime))
    if force_summary:
        lines.extend(force_table(force_summary))
    if cavitation:
        lines.extend(cavitation_table(cavitation))
    if optimization:
        lines.extend(optimization_table(optimization))
    if hso_validation:
        lines.extend(hso_validation_table(hso_validation, args.data_dir))
    lines.extend(recommended_artifacts())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"[ok] wrote {args.output}")


def status_section(audit, fields, runtime, forces, cavitation, optimization):
    items = {
        "Corrected dataset audit": bool(audit and audit.get("valid_for_AoA_or_shape_optimization")),
        "Held-out field metrics": bool(fields),
        "Runtime benchmark": bool(runtime),
        "Model force validation": bool(forces),
        "Cavitation-risk evaluation": bool(cavitation),
        "Continuous HSO results": bool(optimization),
    }
    lines = ["## Evidence Status", "", "| Artifact | Status |", "|---|---|"]
    lines.extend(f"| {name} | {'ready' if ready else 'missing'} |" for name, ready in items.items())
    lines.append("")
    return lines


def field_table(rows):
    selected = [row for row in rows if row.get("field") in {"p", "Cp", "Ux", "Uy", "Cl", "Cd"}]
    lines = ["## Held-Out Field Accuracy", "", "| Model | Field | RMSE | MAE | R2 |", "|---|---|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['model']} | {row['field']} | {f(row.get('rmse'))} | {f(row.get('mae'))} | {f(row.get('r2'))} |")
    return lines + [""]


def runtime_table(rows):
    lines = ["## Runtime", "", "| Model | Inference (s/case) | OpenFOAM (s/case) | Speedup |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {f(row.get('median_inference_seconds_per_case'))} | {f(row.get('median_openfoam_execution_seconds_per_case'))} | {f(row.get('speedup_vs_openfoam'))}x |")
    return lines + [""]


def training_table(training):
    lines = ["## Training Cost", "", "| Model | Parameters | Best epoch | Training time (s) | Best validation objective |", "|---|---:|---:|---:|---:|"]
    for model, values in training.items():
        if not values:
            continue
        lines.append(f"| {model} | {values['parameter_count']} | {values['best_epoch']} | {f(values['training_seconds'])} | {f(values['best_val_loss'])} |")
    return lines + [""]


def force_table(summary):
    lines = ["## Force Accuracy", "", "| Model | Cl RMSE vs OpenFOAM | Cd RMSE vs OpenFOAM |", "|---|---:|---:|"]
    for model, values in summary.items():
        lines.append(f"| {model} | {f(values.get('Cl_vs_openfoam_total', {}).get('rmse'))} | {f(values.get('Cd_vs_openfoam_total', {}).get('rmse'))} |")
    return lines + [""]


def cavitation_table(summary):
    lines = ["## Incipient Cavitation Risk", "", "| Model | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
    for model, values in summary.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"| {model} | {f(values.get('precision'))} | {f(values.get('recall'))} | {f(values.get('f1'))} |")
    return lines + ["", "This is pressure-threshold onset risk from single-phase RANS Cp, not multiphase cavity evolution.", ""]


def optimization_table(rows):
    lines = ["## Shape Optimization", "", "| Model | Start | m | p | t | AoA | Best L/D | Min margin (Pa) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['start']} | {f(row.get('best_m'))} | {f(row.get('best_p'))} | {f(row.get('best_t'))} | {f(row.get('best_AoA'))} | {f(row.get('best_L_over_D'))} | {f(row.get('best_min_cavitation_margin'))} |")
    return lines + [""]


def hso_validation_table(rows, data_dir):
    import numpy as np

    baselines = {}
    for path in data_dir.glob("case_*_grid.npz"):
        with np.load(path, allow_pickle=True) as data:
            key = (str(data["naca"]), float(data["Re"]))
            cl, cd = float(data["Cl_openfoam"]), float(data["Cd_openfoam"])
            if abs(cl) > 5 or abs(cd) > 5 or cd <= 0:
                continue
            baselines[key] = max(baselines.get(key, float("-inf")), cl / cd)
    lines = ["## OpenFOAM-Revalidated Optimization", "", "| Model | Baseline | Predicted L/D | OpenFOAM L/D | Improvement vs baseline |", "|---|---|---:|---:|---:|"]
    for row in sorted(rows, key=lambda item: item["model"]):
        baseline = baselines.get((row["start"], 500000.0), float("nan"))
        achieved = float(row["openfoam_L_over_D"])
        improvement = 100.0 * (achieved / baseline - 1.0)
        lines.append(f"| {row['model']} | NACA {row['start']} ({f(baseline)}) | {f(row['predicted_L_over_D'])} | {f(achieved)} | {f(improvement)}% |")
    return lines + [""]


def recommended_artifacts():
    return [
        "## Paper Figures and Tables",
        "",
        "1. Dataset/design-space table: NACA families, Reynolds numbers, AoA values, grid size, train/validation split.",
        "2. Training and validation loss curves for all four models.",
        "3. Pressure and Cp truth/prediction/error maps on interpolation and held-out-geometry cases.",
        "4. Field accuracy table with RMSE, MAE, R2, parameter count, and training time.",
        "5. OpenFOAM versus surrogate runtime and speedup table.",
        "6. Cl/Cd parity plots against direct OpenFOAM forceCoeffs.",
        "7. Cavitation-risk precision/recall/F1 and representative margin maps.",
        "8. HSO convergence curves, optimized profiles, cross-model ranking agreement, and OpenFOAM revalidation of winners.",
        "9. Oval and random-shape out-of-distribution stress-test table, reported separately from in-distribution optimization.",
        "",
    ]


def read_csv(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def f(value):
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
