#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate optimized endpoints over training seeds.")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "hso_results" / "revised")
    args = parser.parse_args()
    rows = []
    for seed_dir in sorted(args.input_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        endpoint = {row["model"]: row for row in csv.DictReader((seed_dir / "openfoam_validation.csv").open())}
        for model, row in endpoint.items():
            rows.append({"seed": seed, "model": model, **row})
    write_csv(args.input_dir / "openfoam_endpoints_multiseed.csv", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summary = []
    for model, values in sorted(grouped.items()):
        result = {"model": model, "n_seeds": len(values)}
        for metric in ("predicted_L_over_D", "openfoam_Cl", "openfoam_Cd", "openfoam_L_over_D"):
            data = np.asarray([float(row[metric]) for row in values])
            result[f"{metric}_mean"] = float(np.mean(data))
            result[f"{metric}_std"] = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
        summary.append(result)
    write_csv(args.input_dir / "openfoam_endpoints_summary.csv", summary)
    print(f"[ok] aggregated {len(rows)} CFD endpoints")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("No HSO endpoint rows found")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
