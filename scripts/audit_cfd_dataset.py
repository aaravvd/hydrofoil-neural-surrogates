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
    parser = argparse.ArgumentParser(description="Audit CFD grids for duplicated conditions and force references.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed_grids")
    parser.add_argument("--foam-dir", type=Path, default=ROOT / "openfoam_cases")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_results" / "dataset_audit.json")
    args = parser.parse_args()

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    paths = sorted(args.data_dir.glob("case_*_grid.npz"))
    positive_cavitation = 0
    for path in paths:
        data = np.load(path, allow_pickle=True)
        record = {
            "case": path.stem.replace("_grid", ""),
            "AoA": float(data["AoA"]),
            "p": data["p"],
        }
        groups[(str(data["naca"]), float(data["Re"]))].append(record)
        positive_cavitation += int(np.nansum(data["cavitation_indicator"]))

    comparisons = []
    exact_duplicates = 0
    for (naca, reynolds), records in groups.items():
        records.sort(key=lambda item: item["AoA"])
        reference = records[0]
        for current in records[1:]:
            max_abs = float(np.nanmax(np.abs(reference["p"] - current["p"])))
            duplicate = max_abs == 0.0 and reference["AoA"] != current["AoA"]
            exact_duplicates += int(duplicate)
            comparisons.append(
                {
                    "naca": naca,
                    "Re": reynolds,
                    "reference_case": reference["case"],
                    "reference_AoA": reference["AoA"],
                    "case": current["case"],
                    "AoA": current["AoA"],
                    "max_abs_pressure_difference_Pa": max_abs,
                    "exact_duplicate_across_AoA": duplicate,
                }
            )

    force_files = list(args.foam_dir.glob("case_*/postProcessing/forceCoeffs/*/forceCoeffs.dat"))
    payload = {
        "grid_case_count": len(paths),
        "naca_re_groups": len(groups),
        "cross_AoA_comparisons": len(comparisons),
        "exact_pressure_duplicates_across_AoA": exact_duplicates,
        "positive_cavitation_cells": positive_cavitation,
        "openfoam_force_coefficient_files": len(force_files),
        "valid_for_AoA_or_shape_optimization": exact_duplicates == 0 and bool(force_files),
        "finding": (
            "AoA-labelled grids contain exact duplicates and no OpenFOAM forceCoeffs references. "
            "Regenerate corrected CFD before force or shape-optimization claims."
            if exact_duplicates or not force_files
            else "No duplicate-AoA pressure fields detected and force references are available."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]) if comparisons else [])
        if comparisons:
            writer.writeheader()
            writer.writerows(comparisons)
    print(json.dumps(payload, indent=2))
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
