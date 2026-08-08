#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparable training and validation loss histories.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs")
    parser.add_argument("--models", default="unet,pinn,fno,deeponet")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_results" / "training_loss_curves.png")
    args = parser.parse_args()

    histories = {}
    rows = []
    for model in [item.strip() for item in args.models.split(",") if item.strip()]:
        path = args.run_dir / model / "metrics.json"
        if not path.exists():
            print(f"[skip] {model}: {path} does not exist")
            continue
        history = json.loads(path.read_text()).get("history", [])
        if not history:
            print(f"[skip] {model}: no history")
            continue
        histories[model] = history
        rows.extend({"model": model, **point} for point in history)

    if not histories:
        raise SystemExit("No training histories found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for model, history in histories.items():
        epochs = [point["epoch"] for point in history]
        axes[0].plot(epochs, [point["train_loss"] for point in history], label=model)
        axes[1].plot(epochs, [point["val_loss"] for point in history], label=model)
    for ax, title in zip(axes, ["Training loss", "Validation loss"]):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized objective")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)

    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {args.output}")
    print(f"[ok] wrote {csv_path}")


if __name__ == "__main__":
    main()
