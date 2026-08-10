#!/usr/bin/env python3
"""Build the compact, reproducible figures used by the AI4S paper."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
SIM_OUT = ROOT / "papers/sim2science26/figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hydrofoil_pipeline.naca import coordinates, coordinates_from_parameters


MODELS = ["unet", "pinn", "fno", "deeponet"]
LABELS = {"unet": "CNN-U-Net", "pinn": "Physics-reg. MLP", "fno": "FNO", "deeponet": "DeepONet"}
COLORS = {"unet": "#2878B5", "pinn": "#D95F02", "fno": "#4E9F3D", "deeponet": "#7B4AB5"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict[str, str]], model: str, field: str, name: str) -> float:
    row = next(row for row in rows if row["model"] == model and row["field"] == field)
    return float(row[name])


def tradeoff_figure() -> None:
    fields = read_csv(ROOT / "paper_results/revised/seed_7/field_metrics.csv")
    runtime = {r["model"]: r for r in read_csv(ROOT / "paper_results/revised/seed_7/runtime_speedup.csv")}
    with (ROOT / "paper_results/revised/seed_7/cavitation_risk/cavitation_risk_summary.json").open() as handle:
        cav = json.load(handle)

    panels = [
        ("Pressure $R^2$", [metric(fields, m, "p", "r2") for m in MODELS], (-0.05, 1.05), ".2f"),
        ("$C_L$ $R^2$", [metric(fields, m, "Cl", "r2") for m in MODELS], (0.85, 1.01), ".3f"),
        ("Cavitation-risk F1", [float(cav[m]["f1"]) for m in MODELS], (0.7, 0.95), ".3f"),
        ("Speedup over OpenFOAM", [float(runtime[m]["speedup_vs_openfoam"]) for m in MODELS], (0, 2000), ".0f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    for ax, (title, values, ylim, fmt) in zip(axes.flat, panels):
        bars = ax.bar(range(len(MODELS)), values, color=[COLORS[m] for m in MODELS], width=0.72)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS], rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, format(value, fmt), ha="center", va="bottom", fontsize=7)
    fig.savefig(OUT / "model_tradeoffs.pdf", bbox_inches="tight")
    fig.savefig(OUT / "model_tradeoffs.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def field_r2_figure() -> None:
    rows = read_csv(ROOT / "paper_results/revised/seed_7/field_metrics.csv")
    fields = ["Ux", "Uy", "p", "Cp", "nut", "k", "omega"]
    field_labels = [r"$U_x$", r"$U_y$", r"$p$", r"$C_p$", r"$\nu_t$", r"$k$", r"$\omega$"]
    values = np.array([[metric(rows, model, field, "r2") for field in fields] for model in MODELS])

    fig, ax = plt.subplots(figsize=(7.2, 2.25), constrained_layout=True)
    image = ax.imshow(values, cmap="RdYlBu", vmin=-0.05, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(fields)), field_labels)
    ax.set_yticks(range(len(MODELS)), [LABELS[model] for model in MODELS])
    ax.tick_params(labelsize=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 0.18 or values[row, column] > 0.82 else "black"
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label(r"Validation $R^2$", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT / "field_r2_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUT / "field_r2_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def optimization_figure() -> None:
    rows = read_csv(ROOT / "hso_results/revised/seed_7/openfoam_validation.csv")
    by_model = {r["model"]: r for r in rows}
    predicted = [float(by_model[m]["predicted_L_over_D"]) for m in MODELS]
    validated = [float(by_model[m]["openfoam_L_over_D"]) for m in MODELS]
    baselines = {"unet": 25.89, "pinn": 25.57, "fno": 25.57, "deeponet": 25.89}
    improvement = [100 * (validated[i] / baselines[m] - 1) for i, m in enumerate(MODELS)]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    x = np.arange(len(MODELS))
    width = 0.36
    axes[0].bar(x - width / 2, predicted, width, label="Surrogate", color="#5B8DB8")
    axes[0].bar(x + width / 2, validated, width, label="OpenFOAM revalidation", color="#E1812C")
    axes[0].set_ylabel("Lift-to-drag ratio")
    axes[0].set_xticks(x, [LABELS[m] for m in MODELS], rotation=18, ha="right", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    bars = axes[1].bar(x, improvement, color=[COLORS[m] for m in MODELS], width=0.7)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("CFD-validated improvement (%)")
    axes[1].set_xticks(x, [LABELS[m] for m in MODELS], rotation=18, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, improvement):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / "optimization_validation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "optimization_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def aso_results_figure() -> None:
    rows = read_csv(ROOT / "hso_results/revised/seed_7/openfoam_validation.csv")
    by_model = {row["model"]: row for row in rows}
    cfd_summaries = read_csv(
        ROOT / "hso_results/corrected/openfoam_baseline/optimization_summary.csv"
    )
    cfd_best = max(cfd_summaries, key=lambda row: float(row["best_objective"]))
    cfd_trace = read_csv(
        ROOT / "hso_results/corrected/openfoam_baseline/optimization_trace.csv"
    )
    baselines = {"unet": 25.8913357988, "pinn": 25.5700378416,
                 "fno": 25.5700378416, "deeponet": 25.8913357988}

    design_rows = [(LABELS[model], by_model[model], COLORS[model]) for model in MODELS]
    design_rows.append(("Direct OpenFOAM", cfd_best, "#222222"))
    reference_x, reference_y = coordinates("2412")

    fig = plt.figure(figsize=(8.2, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 10, height_ratios=[1.15, 1.55])
    for index, (label, row, color) in enumerate(design_rows):
        ax = fig.add_subplot(grid[0, 2 * index:2 * index + 2])
        if label == "Direct OpenFOAM":
            m, p, thickness = (
                float(row["best_m"]), float(row["best_p"]), float(row["best_t"])
            )
        else:
            m, p, thickness = float(row["m"]), float(row["p"]), float(row["t"])
        opt_x, opt_y = coordinates_from_parameters(m, p, thickness)
        ax.plot(reference_x, reference_y, color="#777777", linestyle="--", linewidth=1.2,
                label="NACA 2412 reference")
        ax.plot(opt_x, opt_y, color=color, linewidth=1.8, label="optimized")
        ax.fill(opt_x, opt_y, color=color, alpha=0.10)
        ax.set_title(f"{label}\nNACA 2412 ref.", fontsize=7.5, pad=2)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.12, 0.14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([-0.1, 0, 0.1] if index == 0 else [])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18, linewidth=0.5)

    ax = fig.add_subplot(grid[1, :])
    x = np.arange(5)
    width = 0.27
    direct_start = next(
        float(row["L_over_D"])
        for row in cfd_trace
        if row["start"] == cfd_best["start"] and int(row["evaluation"]) == 1
    )
    baseline_values = [baselines[model] for model in MODELS] + [direct_start]
    predicted = [float(by_model[model]["predicted_L_over_D"]) for model in MODELS]
    validated = [float(by_model[model]["openfoam_L_over_D"]) for model in MODELS] + [
        float(cfd_best["best_L_over_D"])
    ]
    ax.bar(x - width, baseline_values, width, label="Initial foil (OpenFOAM)", color="#A7A7A7")
    ax.bar(x[:4], predicted, width, label="Surrogate prediction", color="#5B8DB8")
    bars = ax.bar(x + width, validated, width, label="OpenFOAM result", color="#E1812C")
    ax.set_ylabel("Lift-to-drag ratio")
    bar_labels = ["CNN-\nU-Net", "Physics-reg.\nMLP", "FNO", "DeepONet", "Direct\nCFD"]
    ax.set_xticks(x, bar_labels, fontsize=7)
    ax.set_ylim(0, max(58, 1.18 * max(baseline_values + predicted + validated)))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.15))
    for bar, value in zip(bars, validated):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0,
                f"{value:.2f}", ha="center", va="bottom", fontsize=6.5)

    fig.savefig(OUT / "aso_results.pdf", bbox_inches="tight")
    fig.savefig(OUT / "aso_results.png", dpi=240, bbox_inches="tight")
    SIM_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(SIM_OUT / "aso_results.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_existing() -> None:
    sources = {
        ROOT / "paper_results/revised/seed_7/training_loss_curves.png": OUT / "training_loss_curves.png",
        ROOT / "paper_results/revised/seed_7/model_forces/model_force_parity.png": OUT / "force_parity.png",
        ROOT / "figures/revised_model_predictions_test/case_179_p_predictions.png": OUT / "heldout_pressure.png",
    }
    for source, destination in sources.items():
        shutil.copy2(source, destination)
    SIM_OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "figures/revised_model_predictions_test/case_179_p_predictions.png",
        SIM_OUT / "heldout_pressure.png",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    tradeoff_figure()
    field_r2_figure()
    optimization_figure()
    aso_results_figure()
    copy_existing()
    print(f"Wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
