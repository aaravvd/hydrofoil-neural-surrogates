#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrofoil_pipeline.cases import load_config, write_metadata
from hydrofoil_pipeline.extract_openfoam import extract_case
from hydrofoil_pipeline.openfoam import openfoam_available, run_openfoam_case, write_case
from hydrofoil_pipeline.plots import plot_case
from hydrofoil_pipeline.postprocess import grid_case
from hydrofoil_pipeline.synthetic import generate_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["analytic_smoke", "openfoam"], default="analytic_smoke")
    parser.add_argument("--openfoam-backend", choices=["host", "docker"], default="docker")
    parser.add_argument("--run-cfd", action="store_true", help="Run simpleFoam after writing case folders.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Number of independent CFD cases to run concurrently.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse cases whose processed grid already exists.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-case field figures during bulk generation.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=ROOT,
        help="Directory for data, figures, and OpenFOAM cases; must be inside the repository for Docker mode.",
    )
    args = parser.parse_args()

    cases, cfg = load_config(args.config)
    if args.limit is not None:
        cases = cases[: args.limit]

    artifact_root = args.artifact_root.resolve()
    if args.openfoam_backend == "docker":
        try:
            artifact_root.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("--artifact-root must be inside the repository when using Docker.") from exc
    raw_dir = artifact_root / "data" / "raw_cases"
    grid_dir = artifact_root / "data" / "processed_grids"
    figures_dir = artifact_root / "figures"
    foam_dir = artifact_root / "openfoam_cases"
    grid_cfg = cfg["grid"]

    def process_case(case):
        case_dir = foam_dir / case.id
        raw_path = raw_dir / f"{case.id}.npz"
        grid_path = grid_dir / f"{case.id}_grid.npz"
        if args.skip_existing and grid_path.exists():
            return f"[reuse] {case.id}: grid={grid_path}"
        write_case(case, case_dir)

        if args.mode == "analytic_smoke":
            generate_case(case, raw_path)
        else:
            use_docker = args.openfoam_backend == "docker"
            if args.run_cfd:
                run_openfoam_case(case_dir, ROOT, use_docker=use_docker)
                extract_case(case, case_dir, raw_path)
            elif not openfoam_available(use_docker=use_docker):
                print(f"[skip] {case.id}: OpenFOAM backend unavailable; wrote case scaffold only.")
                return f"[skip] {case.id}: backend unavailable"
            else:
                print(f"[skip] {case.id}: use --run-cfd to execute simpleFoam.")
                return f"[skip] {case.id}: --run-cfd not set"

        grid_case(raw_path, grid_path, grid_cfg)
        if not args.no_plots:
            plot_case(grid_path, figures_dir)
        return f"[ok] {case.id}: raw={raw_path} grid={grid_path}"

    if args.jobs <= 1:
        for case in cases:
            print(process_case(case))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(process_case, case): case for case in cases}
            for future in as_completed(futures):
                print(future.result())

    metadata_path = artifact_root / "data" / "metadata.csv"
    write_metadata(cases, metadata_path, args.mode)
    print(f"[done] metadata={metadata_path}")


if __name__ == "__main__":
    main()
