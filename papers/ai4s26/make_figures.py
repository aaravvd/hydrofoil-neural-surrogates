#!/usr/bin/env python3
"""Build the compact, reproducible figures used by the AI4S paper."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    fields = read_csv(ROOT / "paper_results/corrected/field_metrics.csv")
    runtime = {r["model"]: r for r in read_csv(ROOT / "paper_results/corrected/runtime_speedup.csv")}
    with (ROOT / "paper_results/corrected/cavitation_risk/cavitation_risk_summary.json").open() as handle:
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


def optimization_figure() -> None:
    rows = read_csv(ROOT / "hso_results/corrected/optimized/openfoam_validation.csv")
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


def copy_existing() -> None:
    sources = {
        ROOT / "paper_results/corrected/training_loss_curves.png": OUT / "training_loss_curves.png",
        ROOT / "paper_results/corrected/model_forces/model_force_parity.png": OUT / "force_parity.png",
        ROOT / "figures/corrected_model_predictions_heldout/case_160_p_predictions.png": OUT / "heldout_pressure.png",
    }
    for source, destination in sources.items():
        shutil.copy2(source, destination)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    tradeoff_figure()
    optimization_figure()
    copy_existing()
    print(f"Wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
