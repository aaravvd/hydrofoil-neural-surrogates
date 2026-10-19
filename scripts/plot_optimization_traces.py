#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["unet", "pinn", "fno", "deeponet"]
LABELS = {"unet": "CNN-U-Net", "pinn": "PINN", "fno": "FNO", "deeponet": "DeepONet"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all multi-start optimization traces over all training seeds.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "hso_results" / "revised")
    parser.add_argument("--output", type=Path, default=ROOT / "hso_results" / "revised" / "optimization_traces.png")
    parser.add_argument(
        "--direct-cfd-trace",
        type=Path,
        default=ROOT / "hso_results" / "corrected" / "openfoam_baseline" / "optimization_trace.csv",
    )
    args = parser.parse_args()
    traces = load_traces(args.input_dir)
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.3), sharex=True)
    colors = {"0012": "#0072B2", "2412": "#D55E00", "4415": "#009E73"}
    seed_styles = {"7": "-", "17": "--", "27": ":"}
    for ax, model in zip(axes.flat, MODEL_ORDER):
        for (seed, start), rows in sorted(traces[model].items()):
            objective = np.asarray([float(row["objective"]) for row in rows])
            best = np.maximum.accumulate(objective)
            ax.plot(
                np.arange(1, len(best) + 1), best,
                color=colors[start], linestyle=seed_styles[seed], alpha=0.72, lw=1.05,
            )
        ax.set_title(LABELS[model], fontsize=9)
        ax.set_ylabel("Best predicted $C_L/|C_D|$")
        ax.grid(alpha=0.22, linewidth=0.5)
    direct_ax = axes.flat[4]
    if args.direct_cfd_trace.exists():
        direct_rows = list(csv.DictReader(args.direct_cfd_trace.open()))
        for start, color in colors.items():
            rows = [row for row in direct_rows if row["start"] == start and row.get("status") == "ok"]
            objective = np.asarray([float(row["objective"]) for row in rows])
            direct_ax.plot(np.arange(1, len(objective) + 1), np.maximum.accumulate(objective), color=color, lw=1.1)
    direct_ax.set_title("Direct OpenFOAM", fontsize=9)
    direct_ax.set_ylabel("Best CFD $C_L/|C_D|$")
    direct_ax.grid(alpha=0.22, linewidth=0.5)
    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    start_handles = [Line2D([0], [0], color=color, lw=1.6, label=f"NACA {start}") for start, color in colors.items()]
    seed_handles = [Line2D([0], [0], color="#333333", linestyle=style, lw=1.6, label=f"Seed {seed}") for seed, style in seed_styles.items()]
    legend_ax.legend(
        handles=start_handles + seed_handles, title="Color: initializer     Style: seed",
        frameon=False, loc="center", ncol=2, fontsize=8, title_fontsize=8,
        columnspacing=1.8, handlelength=2.5,
    )
    for ax in [axes[1, 0], axes[1, 1], direct_ax]:
        ax.set_xlabel("Objective evaluations")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {args.output}")


def load_traces(root: Path):
    grouped = {model: defaultdict(list) for model in MODEL_ORDER}
    for path in sorted(root.glob("seed_*/optimization_trace.csv")):
        seed = path.parent.name.split("_")[-1]
        for row in csv.DictReader(path.open()):
            grouped[row["model"]][(seed, row["start"])].append(row)
    if not any(grouped[model] for model in MODEL_ORDER):
        raise SystemExit(f"No traces found under {root}")
    return grouped


if __name__ == "__main__":
    main()
