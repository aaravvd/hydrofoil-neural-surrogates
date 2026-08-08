#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def solver_convergence(foam_dir: Path) -> dict:
    final_residuals = []
    cl_spans = []
    cd_spans = []
    retained_cases = set()
    reached_end = 0
    for force_path in sorted(foam_dir.glob("case_*/postProcessing/forceCoeffs/*/forceCoeffs.dat")):
        values = np.loadtxt(force_path, comments="#", ndmin=2)
        if values.shape[1] < 4 or abs(values[-1, 2]) > 5 or abs(values[-1, 3]) > 5:
            continue
        retained_cases.add(force_path.parents[3].name)
        tail = values[values[:, 0] >= values[-1, 0] - 50]
        if tail.shape[0] >= 2:
            cd_spans.append(float(np.ptp(tail[:, 2])))
            cl_spans.append(float(np.ptp(tail[:, 3])))

    logs = sorted(foam_dir.glob("case_*/log.simpleFoam"))
    for log_path in logs:
        if log_path.parent.name not in retained_cases:
            continue
        text = log_path.read_text(errors="replace")
        marker = text.rfind("Time = 300")
        if marker >= 0:
            reached_end += 1
            block = text[marker:text.find("ExecutionTime", marker)]
            residuals = [
                float(value)
                for value in re.findall(r"Initial residual = ([0-9.eE+\-]+)", block)
            ]
            if residuals:
                final_residuals.append(max(residuals))

    def distribution(values: list[float]) -> dict:
        return {
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "maximum": float(np.max(values)) if values else None,
        }

    return {
        "retained_case_count": len(retained_cases),
        "cases_reaching_iteration_300": reached_end,
        "max_initial_residual_at_iteration_300": distribution(final_residuals),
        "Cl_span_over_final_50_iterations": distribution(cl_spans),
        "Cd_span_over_final_50_iterations": distribution(cd_spans),
    }


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
        "solver_convergence": solver_convergence(args.foam_dir),
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
