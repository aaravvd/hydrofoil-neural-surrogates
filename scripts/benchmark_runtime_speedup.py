#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.visualize_model_predictions import choose_cases, load_checkpoints, predict_case, require_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark surrogate inference time and compare with OpenFOAM logs.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed_grids")
    parser.add_argument("--openfoam-dir", type=Path, default=ROOT / "openfoam_cases")
    parser.add_argument("--output", type=Path, default=ROOT / "paper_results" / "runtime_speedup.csv")
    parser.add_argument("--models", default="unet,fno,deeponet,pinn")
    parser.add_argument("--num-cases", type=int, default=20)
    parser.add_argument("--split", choices=["test", "validation", "train", "all"], default="test")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs" / "family_split.json")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.case = []

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = [m.strip() for m in args.models.split(",") if m.strip()]
    checkpoints = load_checkpoints(torch, args.run_dir, selected, device)
    case_paths = choose_cases(args)
    rows = []
    cfd_times = [parse_openfoam_time(args.openfoam_dir / path.name.replace("_grid.npz", "") / "log.simpleFoam") for path in case_paths]
    cfd_times = [t for t in cfd_times if t is not None]
    if not cfd_times:
        cfd_times = [t for t in (parse_openfoam_time(path) for path in args.openfoam_dir.glob("case_*/log.simpleFoam")) if t is not None]
    median_cfd = sorted(cfd_times)[len(cfd_times) // 2] if cfd_times else None

    for model_name, bundle in checkpoints.items():
        timings = []
        for _ in range(args.warmup):
            for path in case_paths[: min(3, len(case_paths))]:
                predict_case(torch, bundle, path, "p", args.batch_size, device)
        for _ in range(args.repeats):
            start = time.perf_counter()
            for path in case_paths:
                predict_case(torch, bundle, path, "p", args.batch_size, device)
            elapsed = time.perf_counter() - start
            timings.append(elapsed / len(case_paths))
        median_inference = sorted(timings)[len(timings) // 2]
        rows.append(
            {
                "model": model_name,
                "cases_timed": len(case_paths),
                "median_inference_seconds_per_case": median_inference,
                "median_openfoam_execution_seconds_per_case": median_cfd if median_cfd is not None else "",
                "speedup_vs_openfoam": (median_cfd / median_inference) if median_cfd is not None else "",
                "note": "OpenFOAM time parsed from log.simpleFoam ExecutionTime; inference predicts pressure field only for timing.",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    print(f"[ok] wrote {args.output}")


def parse_openfoam_time(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    matches = re.findall(r"ExecutionTime\s*=\s*([0-9.]+)\s*s", text)
    return float(matches[-1]) if matches else None


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
