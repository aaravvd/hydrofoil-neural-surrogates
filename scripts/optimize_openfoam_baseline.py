#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydrofoil_pipeline.cases import Case
from hydrofoil_pipeline.extract_openfoam import latest_time_dir, read_internal_field
from hydrofoil_pipeline.naca import coordinates_from_parameters, parse_naca4
from hydrofoil_pipeline.openfoam import run_openfoam_case, write_case
from scripts.validate_force_calculator import read_force_coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct OpenFOAM baseline for the continuous NACA shape-optimization experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hso_results" / "corrected" / "openfoam_baseline")
    parser.add_argument("--starts", default="0012,2412,4415")
    parser.add_argument("--Re", type=float, default=500000.0)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--rho", type=float, default=997.0)
    parser.add_argument("--nu", type=float, default=1e-6)
    parser.add_argument("--p-inf", type=float, default=101325.0)
    parser.add_argument("--p-vap", type=float, default=2300.0)
    parser.add_argument("--chord", type=float, default=1.0)
    parser.add_argument("--cavitation-margin-threshold", type=float, default=5000.0)
    parser.add_argument(
        "--import-dir", type=Path, action="append", default=[],
        help="Import cached trace rows from another baseline directory before optimizing.",
    )
    parser.add_argument(
        "--summarize-only", action="store_true",
        help="Write summaries from successful cached evaluations without running the optimizer.",
    )
    parser.add_argument(
        "--drop-failed", action="store_true",
        help="Remove failed rows from the cached trace before summarizing.",
    )
    parser.add_argument("--host-openfoam", action="store_true", help="Use host OpenFOAM instead of Docker.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cache_key(start: str, values: np.ndarray) -> str:
    payload = start + ":" + ":".join(f"{float(value):.10f}" for value in values)
    return hashlib.sha1(payload.encode("ascii")).hexdigest()[:12]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "optimization_trace.csv"
    summary_path = args.output_dir / "optimization_summary.csv"
    traces = list(csv.DictReader(trace_path.open())) if trace_path.exists() else []
    known_keys = {row["cache_key"] for row in traces}
    for import_dir in args.import_dir:
        import_trace = import_dir / "optimization_trace.csv"
        if not import_trace.exists():
            raise FileNotFoundError(import_trace)
        for row in csv.DictReader(import_trace.open()):
            if row["cache_key"] not in known_keys:
                traces.append(row)
                known_keys.add(row["cache_key"])
    if args.import_dir:
        write_csv(trace_path, traces)
    if args.drop_failed:
        traces = [row for row in traces if row.get("status") == "ok"]
        write_csv(trace_path, traces)
    cached = {row["cache_key"]: row for row in traces if row.get("status") == "ok"}
    summaries = list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
    start_codes = [item.strip() for item in args.starts.split(",") if item.strip()]

    if args.summarize_only:
        for start_code in start_codes:
            candidates = [
                row for row in traces if row.get("start") == start_code and row.get("status") == "ok"
            ]
            if not candidates:
                raise RuntimeError(f"No successful cached evaluations for NACA {start_code}")
            best = max(candidates, key=lambda row: float(row["objective"]))
            summary = {
                "start": start_code,
                "success": False,
                "message": "Best observed within the fixed iteration budget.",
                "evaluations": len(candidates),
                "best_m": best["m"],
                "best_p": best["p"],
                "best_t": best["t"],
                "best_AoA": best["AoA"],
                "best_Cl": best["Cl"],
                "best_Cd": best["Cd"],
                "best_L_over_D": best["L_over_D"],
                "best_min_cavitation_margin": best["min_cavitation_margin"],
                "best_objective": best["objective"],
                "best_case_id": best["case_id"],
            }
            summaries = [row for row in summaries if row.get("start") != start_code]
            summaries.append(summary)
        summaries.sort(key=lambda row: int(row["start"]))
        write_csv(summary_path, summaries)
        (args.output_dir / "optimization_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        print(f"[ok] wrote {summary_path}")
        return

    for start_code in start_codes:
        foil = parse_naca4(start_code)
        x0 = np.array([foil.m, max(foil.p, 0.4), foil.t, 4.0], dtype=float)
        evaluation = 0

        def objective(values: np.ndarray) -> float:
            nonlocal evaluation
            evaluation += 1
            values = np.asarray(values, dtype=float)
            key = cache_key(start_code, values)
            if key in cached:
                return -float(cached[key]["objective"])

            m, p, thickness, aoa = map(float, values)
            case_id = f"start_{start_code}_eval_{evaluation:03d}_{key}"
            case = Case(
                id=case_id,
                naca="custom",
                AoA=aoa,
                Re=args.Re,
                rho=args.rho,
                nu=args.nu,
                p_inf=args.p_inf,
                p_vap=args.p_vap,
                chord=args.chord,
            )
            case_dir = args.output_dir / "openfoam_cases" / case_id
            coords = coordinates_from_parameters(m, p, thickness, n=121, chord=args.chord)
            started = time.perf_counter()
            row = {
                "cache_key": key,
                "start": start_code,
                "evaluation": evaluation,
                "m": m,
                "p": p,
                "t": thickness,
                "AoA": aoa,
                "case_id": case_id,
            }
            try:
                write_case(case, case_dir, airfoil_coords=coords)
                run_openfoam_case(case_dir, ROOT, use_docker=not args.host_openfoam)
                force = read_force_coefficients(case_dir)
                if force is None or not np.isfinite(force["Cl"]) or not np.isfinite(force["Cd"]):
                    raise RuntimeError("OpenFOAM did not produce finite force coefficients")
                time_dir = latest_time_dir(case_dir)
                p_kinematic = read_internal_field(time_dir / "p")
                margin = args.p_inf + args.rho * p_kinematic - args.p_vap
                min_margin = float(np.nanmin(margin))
                risky_fraction = float(np.nanmean(margin < args.cavitation_margin_threshold))
                cavitation_penalty = max(
                    0.0,
                    (args.cavitation_margin_threshold - min_margin)
                    / max(args.cavitation_margin_threshold, 1e-9),
                )
                lift_to_drag = float(force["Cl"] / (abs(force["Cd"]) + 1e-6))
                score = lift_to_drag - 10.0 * cavitation_penalty - 5.0 * risky_fraction
                row.update(
                    {
                        "status": "ok",
                        "Cl": force["Cl"],
                        "Cd": force["Cd"],
                        "L_over_D": lift_to_drag,
                        "min_cavitation_margin": min_margin,
                        "risky_cell_fraction": risky_fraction,
                        "objective": score,
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                cached[key] = row
                print(
                    f"[ok] {start_code} eval {evaluation}: L/D={lift_to_drag:.4f}, "
                    f"theta=({m:.4f}, {p:.4f}, {thickness:.4f}, {aoa:.3f})",
                    flush=True,
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "objective": -1e6,
                        "error": str(exc),
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                print(f"[failed] {start_code} eval {evaluation}: {exc}", flush=True)
            traces.append(row)
            write_csv(trace_path, traces)
            return -float(row["objective"])

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            bounds=[(0.0, 0.09), (0.1, 0.9), (0.06, 0.20), (-4.0, 10.0)],
            options={"maxiter": args.maxiter, "xatol": 5e-4, "fatol": 1e-3},
        )
        candidates = [
            row for row in traces if row.get("start") == start_code and row.get("status") == "ok"
        ]
        if not candidates:
            raise RuntimeError(f"No successful OpenFOAM evaluations for NACA {start_code}")
        best = max(candidates, key=lambda row: float(row["objective"]))
        summary = {
            "start": start_code,
            "success": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "best_m": best["m"],
            "best_p": best["p"],
            "best_t": best["t"],
            "best_AoA": best["AoA"],
            "best_Cl": best["Cl"],
            "best_Cd": best["Cd"],
            "best_L_over_D": best["L_over_D"],
            "best_min_cavitation_margin": best["min_cavitation_margin"],
            "best_objective": best["objective"],
            "best_case_id": best["case_id"],
        }
        summaries = [row for row in summaries if row.get("start") != start_code]
        summaries.append(summary)
        summaries.sort(key=lambda row: int(row["start"]))
        write_csv(summary_path, summaries)
        (args.output_dir / "optimization_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    print(f"[ok] wrote {summary_path}")


if __name__ == "__main__":
    main()
