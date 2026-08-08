#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrofoil_pipeline.cases import load_config
from hydrofoil_pipeline.extract_openfoam import extract_case
from hydrofoil_pipeline.postprocess import grid_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-extract completed OpenFOAM cases and rebuild ML grids.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases, config = load_config(args.config)
    if args.limit is not None:
        cases = cases[: args.limit]
    raw_dir = args.artifact_root / "data" / "raw_cases"
    grid_dir = args.artifact_root / "data" / "processed_grids"
    foam_dir = args.artifact_root / "openfoam_cases"
    completed = 0
    for case in cases:
        case_dir = foam_dir / case.id
        if not (case_dir / "log.simpleFoam").exists():
            print(f"[skip] {case.id}: no solver log")
            continue
        raw_path = raw_dir / f"{case.id}.npz"
        grid_path = grid_dir / f"{case.id}_grid.npz"
        extract_case(case, case_dir, raw_path)
        grid_case(raw_path, grid_path, config["grid"])
        completed += 1
        if completed == 1 or completed % 25 == 0:
            print(f"[grid] rebuilt {completed} cases")
    print(f"[done] rebuilt {completed}/{len(cases)} cases")


if __name__ == "__main__":
    main()
