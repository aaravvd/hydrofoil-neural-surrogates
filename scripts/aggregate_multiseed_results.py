#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final-test metrics over fixed-split training seeds.")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "paper_results" / "revised")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs" / "revised")
    args = parser.parse_args()

    rows: list[dict] = []
    for seed_dir in sorted(args.result_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        add_field_rows(rows, seed_dir / "field_metrics.csv", seed)
        add_force_rows(rows, seed_dir / "model_forces" / "model_force_summary.json", seed)
        add_cavitation_rows(rows, seed_dir / "cavitation_risk" / "cavitation_risk_summary.json", seed)
        add_runtime_rows(rows, seed_dir / "runtime_speedup.csv", seed)
        add_training_rows(rows, args.run_dir / seed_dir.name, seed)
    if not rows:
        raise SystemExit(f"No per-seed results found in {args.result_dir}")

    write_csv(args.result_dir / "multiseed_metrics_long.csv", rows)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["metric"])].append(float(row["value"]))
    summary = []
    for (model, metric), values in sorted(grouped.items()):
        summary.append(
            {
                "model": model,
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    write_csv(args.result_dir / "multiseed_metrics_summary.csv", summary)
    print(f"[ok] aggregated {len(rows)} measurements over {len({row['seed'] for row in rows})} seeds")


def add_field_rows(rows: list[dict], path: Path, seed: int) -> None:
    if not path.exists():
        return
    for row in csv.DictReader(path.open()):
        if row["field"] in {"p", "Cp"}:
            for statistic in ("mae", "rmse", "r2"):
                rows.append(record(seed, row["model"], f"{row['field']}_{statistic}", row[statistic]))


def add_force_rows(rows: list[dict], path: Path, seed: int) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    for model, metrics in payload.items():
        for coefficient in ("Cl", "Cd"):
            key = f"{coefficient}_vs_openfoam_total"
            for statistic in ("mae", "rmse", "pearson_r"):
                rows.append(record(seed, model, f"{coefficient}_{statistic}", metrics[key][statistic]))


def add_runtime_rows(rows: list[dict], path: Path, seed: int) -> None:
    if not path.exists():
        return
    for row in csv.DictReader(path.open()):
        rows.append(record(seed, row["model"], "inference_seconds", row["median_inference_seconds_per_case"]))
        rows.append(record(seed, row["model"], "speedup_vs_openfoam", row["speedup_vs_openfoam"]))


def add_cavitation_rows(rows: list[dict], path: Path, seed: int) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    for model, metrics in payload.items():
        if model == "interpretation":
            continue
        for statistic in ("precision", "recall", "f1"):
            rows.append(record(seed, model, f"cavitation_{statistic}", metrics[statistic]))


def add_training_rows(rows: list[dict], run_dir: Path, seed: int) -> None:
    for path in run_dir.glob("*/metrics.json"):
        payload = json.loads(path.read_text())
        model = payload["model"]
        rows.append(record(seed, model, "training_seconds", payload["training_seconds"]))
        rows.append(record(seed, model, "parameter_count", payload["parameter_count"]))


def record(seed: int, model: str, metric: str, value) -> dict:
    return {"seed": seed, "model": model, "metric": metric, "value": float(value)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
