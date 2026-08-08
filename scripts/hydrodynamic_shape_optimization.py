#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from hydrofoil_pipeline.naca import coordinates, geometry_descriptors
from models.architectures import build_model
from models.datasets import denormalize_output_grid, load_normalization


MODEL_ORDER = ["unet", "fno", "deeponet", "pinn", "dnn"]


@dataclass
class Candidate:
    candidate_id: str
    family: str
    airfoil_x: np.ndarray
    airfoil_y: np.ndarray
    AoA: float
    Re: float
    rho: float
    nu: float
    p_inf: float
    p_vap: float
    chord: float

    @property
    def U_inf(self) -> float:
        return self.Re * self.nu / self.chord


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for HSO screening. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-ml.txt"
        ) from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Surrogate-assisted hydrodynamic shape optimization/screening.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "hso_results")
    parser.add_argument("--models", default="auto", help="Comma list from unet,fno,deeponet,pinn,dnn or auto.")
    parser.add_argument("--primary-model", default="auto", help="Model used for paper headline rankings; auto prefers unet then fno.")
    parser.add_argument("--n-random-airfoils", type=int, default=5)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--Re", type=float, default=500000.0)
    parser.add_argument("--aoa-values", default="-2,0,2,4,6,8")
    parser.add_argument("--naca-codes", default="0008,0012,2412,4412,6409,8412")
    parser.add_argument("--grid-nx", type=int, default=128)
    parser.add_argument("--grid-ny", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--cavitation-margin-threshold", type=float, default=5000.0)
    parser.add_argument("--rho", type=float, default=997.0)
    parser.add_argument("--nu", type=float, default=1.0e-6)
    parser.add_argument("--p-inf", type=float, default=101325.0)
    parser.add_argument("--p-vap", type=float, default=2300.0)
    parser.add_argument("--chord", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = MODEL_ORDER if args.models == "auto" else [m.strip() for m in args.models.split(",") if m.strip()]
    bundles = load_checkpoints(torch, args.run_dir, selected, device)
    if not bundles:
        raise SystemExit(f"No usable checkpoints found in {args.run_dir}")

    primary = choose_primary_model(args.primary_model, bundles)
    grid = make_grid(args.grid_nx, args.grid_ny)
    candidates = build_candidates(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_by_model = {}
    for model_name, bundle in bundles.items():
        print(f"[hso] scoring {len(candidates)} candidates with {model_name}")
        model_rows = []
        for candidate in candidates:
            pred = predict_candidate(torch, bundle, candidate, grid, args.batch_size, device)
            row = score_candidate(model_name, candidate, pred, grid, args.cavitation_margin_threshold)
            rows.append(row)
            model_rows.append(row)
        best_by_model[model_name] = sorted(model_rows, key=lambda r: r["objective"], reverse=True)[:10]

    write_csv(args.output_dir / "hso_candidate_scores.csv", rows)
    write_geometry_files(args.output_dir, candidates)
    write_report(args.output_dir / "hso_report.md", args, primary, best_by_model)
    plot_top_geometries(args.output_dir / "top_geometries.png", candidates, best_by_model[primary][:5])
    print(f"[ok] wrote {args.output_dir / 'hso_candidate_scores.csv'}")
    print(f"[ok] wrote {args.output_dir / 'hso_report.md'}")
    print(f"[ok] wrote {args.output_dir / 'top_geometries.png'}")


def load_checkpoints(torch, run_dir: Path, selected: list[str], device) -> dict[str, dict]:
    bundles = {}
    for model_name in selected:
        normalized_name = "unet" if model_name == "cnn_unet" else model_name
        ckpt_path = run_dir / normalized_name / "best.pt"
        norm_path = run_dir / normalized_name / "normalization.npz"
        if not ckpt_path.exists() or not norm_path.exists():
            print(f"[skip] {normalized_name}: missing checkpoint or normalization")
            continue
        ckpt = load_local_checkpoint(torch, ckpt_path, device)
        norm = load_normalization(norm_path)
        train_args = SimpleNamespace(**ckpt.get("args", {}))
        for key, value in {"width": 128, "depth": 4, "basis": 128, "modes": 16}.items():
            if not hasattr(train_args, key):
                setattr(train_args, key, value)
        model = build_model(normalized_name, len(norm.input_fields), len(norm.target_fields), train_args).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        bundles[normalized_name] = {"model": model, "norm": norm}
    return bundles


def load_local_checkpoint(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def choose_primary_model(primary_arg: str, bundles: dict[str, dict]) -> str:
    if primary_arg != "auto":
        if primary_arg not in bundles:
            raise SystemExit(f"Primary model {primary_arg!r} is not loaded. Loaded: {list(bundles)}")
        return primary_arg
    for name in ["unet", "fno", "deeponet", "pinn", "dnn"]:
        if name in bundles:
            return name
    return next(iter(bundles))


def make_grid(nx: int, ny: int) -> dict[str, np.ndarray]:
    gx = np.linspace(-1.0, 2.0, nx, dtype=np.float32)
    gy = np.linspace(-0.75, 0.75, ny, dtype=np.float32)
    x, y = np.meshgrid(gx, gy)
    return {"grid_x": x.astype(np.float32), "grid_y": y.astype(np.float32)}


def build_candidates(args) -> list[Candidate]:
    aoa_values = [float(v) for v in args.aoa_values.split(",") if v.strip()]
    candidates = []
    for code in [v.strip() for v in args.naca_codes.split(",") if v.strip()]:
        ax, ay = coordinates(code, n=241, chord=args.chord)
        for aoa in aoa_values:
            candidates.append(make_candidate(f"naca_{code}_aoa_{aoa:g}", f"NACA {code}", ax, ay, aoa, args))

    rng = np.random.default_rng(args.seed)
    for i in range(1, args.n_random_airfoils + 1):
        ax, ay = random_airfoil(rng, n=241, chord=args.chord)
        for aoa in aoa_values:
            candidates.append(make_candidate(f"random_{i:02d}_aoa_{aoa:g}", f"random_{i:02d}", ax, ay, aoa, args))
    ax, ay = oval_airfoil(n=241, chord=args.chord)
    for aoa in aoa_values:
        candidates.append(make_candidate(f"oval_aoa_{aoa:g}", "oval", ax, ay, aoa, args))
    return candidates


def oval_airfoil(n: int = 241, chord: float = 1.0, thickness_ratio: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    x = 0.5 * chord + 0.5 * chord * np.cos(theta)
    y = 0.5 * thickness_ratio * chord * np.sin(theta)
    return x.astype(np.float32), y.astype(np.float32)


def make_candidate(candidate_id: str, family: str, ax: np.ndarray, ay: np.ndarray, aoa: float, args) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        family=family,
        airfoil_x=ax.astype(np.float32),
        airfoil_y=ay.astype(np.float32),
        AoA=aoa,
        Re=args.Re,
        rho=args.rho,
        nu=args.nu,
        p_inf=args.p_inf,
        p_vap=args.p_vap,
        chord=args.chord,
    )


def random_airfoil(rng: np.random.Generator, n: int, chord: float) -> tuple[np.ndarray, np.ndarray]:
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    base_t = rng.uniform(0.07, 0.16)
    yt = 5.0 * base_t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    yt *= 1.0 + rng.uniform(-0.18, 0.18) * np.sin(math.pi * x) + rng.uniform(-0.10, 0.10) * np.sin(2.0 * math.pi * x)
    yt = np.maximum(yt, 0.006 * np.sin(math.pi * x))
    camber = (
        rng.uniform(-0.025, 0.045) * np.sin(math.pi * x)
        + rng.uniform(-0.018, 0.018) * np.sin(2.0 * math.pi * x)
        + rng.uniform(-0.010, 0.010) * np.sin(3.0 * math.pi * x)
    )
    taper = np.sin(math.pi * x) ** 0.35
    yu = (camber + yt * taper) * chord
    yl = (camber - yt * taper) * chord
    xu = x * chord
    xl = x * chord
    x_closed = np.concatenate([xu[::-1], xl[1:]])
    y_closed = np.concatenate([yu[::-1], yl[1:]])
    y_closed[0] = 0.0
    y_closed[-1] = 0.0
    return x_closed.astype(np.float32), y_closed.astype(np.float32)


def predict_candidate(torch, bundle: dict, candidate: Candidate, grid: dict[str, np.ndarray], batch_size: int, device) -> dict[str, np.ndarray]:
    norm = bundle["norm"]
    x_input, mask = candidate_input_grid(candidate, grid, norm.input_fields)
    x_norm = (x_input - norm.input_mean[:, None, None]) / norm.input_std[:, None, None]
    model = bundle["model"]
    with torch.no_grad():
        if model.__class__.__name__ in {"FNO2d", "UNet2d"}:
            pred = model(torch.from_numpy(x_norm[None]).to(device)).cpu().numpy()[0]
        else:
            channels, height, width = x_norm.shape
            flat = np.moveaxis(x_norm, 0, -1).reshape(height * width, channels)
            chunks = []
            for start in range(0, flat.shape[0], batch_size):
                chunks.append(model(torch.from_numpy(flat[start : start + batch_size]).to(device)).cpu().numpy())
            pred = np.moveaxis(np.concatenate(chunks, axis=0).reshape(height, width, -1), -1, 0)
    pred = denormalize_output_grid(pred, norm)
    for idx in norm.binary_indices:
        pred[idx] = 1.0 / (1.0 + np.exp(-np.clip(pred[idx], -60.0, 60.0)))
    return {field: np.where(mask, pred[i], np.nan) for i, field in enumerate(norm.target_fields)}


def candidate_input_grid(candidate: Candidate, grid: dict[str, np.ndarray], input_fields: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = grid["grid_x"]
    y = grid["grid_y"]
    sdf = signed_distance(x, y, candidate.airfoil_x, candidate.airfoil_y).astype(np.float32)
    mask = sdf > 0.0
    max_camber, camber_position, thickness_ratio = geometry_descriptors(
        candidate.airfoil_x, candidate.airfoil_y, candidate.chord
    )
    values = {
        "x": x,
        "y": y,
        "Re": np.full_like(x, candidate.Re, dtype=np.float32),
        "AoA": np.full_like(x, candidate.AoA, dtype=np.float32),
        "U_inf": np.full_like(x, candidate.U_inf, dtype=np.float32),
        "rho": np.full_like(x, candidate.rho, dtype=np.float32),
        "nu": np.full_like(x, candidate.nu, dtype=np.float32),
        "p_inf": np.full_like(x, candidate.p_inf, dtype=np.float32),
        "p_vap": np.full_like(x, candidate.p_vap, dtype=np.float32),
        "max_camber": np.full_like(x, max_camber, dtype=np.float32),
        "camber_position": np.full_like(x, camber_position, dtype=np.float32),
        "thickness_ratio": np.full_like(x, thickness_ratio, dtype=np.float32),
        "fluid_mask": mask.astype(np.float32),
        "sdf": sdf,
    }
    arr = np.stack([values[name] for name in input_fields], axis=0).astype(np.float32)
    arr = np.where(mask[None], arr, 0.0)
    return arr, mask


def signed_distance(x: np.ndarray, y: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    inside = points_inside_polygon(x, y, ax, ay)
    dist = distance_to_polyline(x, y, ax, ay)
    return np.where(inside, -dist, dist)


def points_inside_polygon(x: np.ndarray, y: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    px = x.ravel()
    py = y.ravel()
    inside = np.zeros_like(px, dtype=bool)
    j = len(ax) - 1
    for i in range(len(ax)):
        intersects = ((ay[i] > py) != (ay[j] > py)) & (
            px < (ax[j] - ax[i]) * (py - ay[i]) / (ay[j] - ay[i] + 1e-15) + ax[i]
        )
        inside ^= intersects
        j = i
    return inside.reshape(x.shape)


def distance_to_polyline(x: np.ndarray, y: np.ndarray, ax: np.ndarray, ay: np.ndarray) -> np.ndarray:
    px = x[..., None]
    py = y[..., None]
    x0 = ax[:-1]
    y0 = ay[:-1]
    x1 = ax[1:]
    y1 = ay[1:]
    dx = x1 - x0
    dy = y1 - y0
    denom = dx * dx + dy * dy + 1e-15
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / denom, 0.0, 1.0)
    qx = x0 + t * dx
    qy = y0 + t * dy
    return np.sqrt(np.min((px - qx) ** 2 + (py - qy) ** 2, axis=-1))


def score_candidate(model_name: str, candidate: Candidate, pred: dict[str, np.ndarray], grid: dict[str, np.ndarray], margin_threshold: float) -> dict:
    p = pred.get("p")
    margin = pred.get("cavitation_margin")
    cp = pred.get("Cp")
    cl_pressure, cd_pressure = pressure_force_coefficients(candidate, grid, p)
    cl_direct = float(np.nanmedian(pred["Cl"])) if "Cl" in pred else float("nan")
    cd_direct = float(np.nanmedian(pred["Cd"])) if "Cd" in pred else float("nan")
    cl = cl_direct if np.isfinite(cl_direct) else cl_pressure
    cd = cd_direct if np.isfinite(cd_direct) else cd_pressure
    min_margin = float(np.nanmin(margin)) if margin is not None else float("nan")
    risky_fraction = float(np.nanmean(margin < margin_threshold)) if margin is not None else float("nan")
    cp_min = float(np.nanmin(cp)) if cp is not None else float("nan")
    cav_penalty = max(0.0, (margin_threshold - min_margin) / max(margin_threshold, 1e-9)) if np.isfinite(min_margin) else 0.0
    objective = cl / (abs(cd) + 1e-6) - 10.0 * cav_penalty - 5.0 * risky_fraction
    return {
        "model": model_name,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "AoA": candidate.AoA,
        "Re": candidate.Re,
        "CL_surrogate": cl,
        "CD_surrogate": cd,
        "CL_integrated_pressure": cl_pressure,
        "CD_integrated_pressure": cd_pressure,
        "force_source": "direct OpenFOAM-coefficient surrogate" if np.isfinite(cl_direct) and np.isfinite(cd_direct) else "pressure integral",
        "CL_over_abs_CD": cl / (abs(cd) + 1e-6),
        "min_cavitation_margin": min_margin,
        "risky_cell_fraction": risky_fraction,
        "min_Cp": cp_min,
        "objective": objective,
        "note": "surrogate estimate; validate top designs with OpenFOAM",
    }


def pressure_force_coefficients(candidate: Candidate, grid: dict[str, np.ndarray], p_abs: np.ndarray) -> tuple[float, float]:
    ax = candidate.airfoil_x
    ay = candidate.airfoil_y
    if ax[0] != ax[-1] or ay[0] != ay[-1]:
        ax = np.r_[ax, ax[0]]
        ay = np.r_[ay, ay[0]]
    area = 0.5 * np.sum(ax[:-1] * ay[1:] - ax[1:] * ay[:-1])
    normal_sign = 1.0 if area > 0.0 else -1.0
    fx = 0.0
    fy = 0.0
    for x0, y0, x1, y1 in zip(ax[:-1], ay[:-1], ax[1:], ay[1:]):
        dx = x1 - x0
        dy = y1 - y0
        ds = math.hypot(float(dx), float(dy))
        if ds <= 1e-12:
            continue
        nx = normal_sign * dy / ds
        ny = -normal_sign * dx / ds
        sample_x = 0.5 * (x0 + x1) + 0.01 * nx
        sample_y = 0.5 * (y0 + y1) + 0.01 * ny
        p = bilinear(grid["grid_x"], grid["grid_y"], p_abs, sample_x, sample_y)
        if not np.isfinite(p):
            continue
        gauge = p - candidate.p_inf
        fx += -gauge * nx * ds
        fy += -gauge * ny * ds
    alpha = math.radians(candidate.AoA)
    drag_axis = np.array([math.cos(alpha), math.sin(alpha)])
    lift_axis = np.array([-math.sin(alpha), math.cos(alpha)])
    force = np.array([fx, fy])
    q = 0.5 * candidate.rho * candidate.U_inf**2 * candidate.chord + 1e-12
    cd = float(force.dot(drag_axis) / q)
    cl = float(force.dot(lift_axis) / q)
    return cl, cd


def bilinear(grid_x: np.ndarray, grid_y: np.ndarray, values: np.ndarray, x: float, y: float) -> float:
    xs = grid_x[0]
    ys = grid_y[:, 0]
    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
        return float("nan")
    ix = int(np.searchsorted(xs, x) - 1)
    iy = int(np.searchsorted(ys, y) - 1)
    ix = int(np.clip(ix, 0, len(xs) - 2))
    iy = int(np.clip(iy, 0, len(ys) - 2))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    q11 = values[iy, ix]
    q21 = values[iy, ix + 1]
    q12 = values[iy + 1, ix]
    q22 = values[iy + 1, ix + 1]
    tx = (x - x0) / (x1 - x0 + 1e-15)
    ty = (y - y0) / (y1 - y0 + 1e-15)
    return float((1 - tx) * (1 - ty) * q11 + tx * (1 - ty) * q21 + (1 - tx) * ty * q12 + tx * ty * q22)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_geometry_files(output_dir: Path, candidates: list[Candidate]) -> None:
    geometry_dir = output_dir / "candidate_geometries"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for candidate in candidates:
        key = candidate.family
        if key in seen:
            continue
        seen.add(key)
        path = geometry_dir / f"{key.replace(' ', '_')}.csv"
        rows = [{"x": float(x), "y": float(y)} for x, y in zip(candidate.airfoil_x, candidate.airfoil_y)]
        write_csv(path, rows)


def write_report(path: Path, args, primary: str, best_by_model: dict[str, list[dict]]) -> None:
    lines = [
        "# Hydrodynamic Shape Optimization Screening Report",
        "",
        "This is surrogate-assisted design-space screening, not final CFD-validated optimization.",
        f"Primary model for headline ranking: `{primary}`.",
        f"Candidates: NACA sweep plus {args.n_random_airfoils} random smooth airfoils across AoA values `{args.aoa_values}` at Re={args.Re:g}.",
        "",
        "Objective: maximize surrogate `CL / abs(CD)` with penalties for cavitation-margin violations.",
        "",
    ]
    for model_name, rows in best_by_model.items():
        lines.append(f"## Top Designs: {model_name}")
        lines.append("")
        lines.append("| Rank | Candidate | Family | AoA | CL surrogate | CD surrogate | CL/abs(CD) | Min cavitation margin | Objective |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for i, row in enumerate(rows, start=1):
            lines.append(
                f"| {i} | {row['candidate_id']} | {row['family']} | {float(row['AoA']):.2f} | "
                f"{float(row['CL_surrogate']):.4g} | {float(row['CD_surrogate']):.4g} | "
                f"{float(row['CL_over_abs_CD']):.4g} | {float(row['min_cavitation_margin']):.4g} | "
                f"{float(row['objective']):.4g} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Paper Caveat",
            "",
            "When checkpoints include direct OpenFOAM `Cl` and `Cd` targets, ranking uses those outputs; older checkpoints "
            "fall back to a pressure-only grid integral. In either case, top candidates should be re-run in OpenFOAM "
            "before making any hydrodynamic performance claim. Random airfoils are out-of-distribution relative to the "
            "current NACA-heavy training set, so they are best presented as a stress test and future validation target.",
            "",
            "## Recommended AI4S Framing",
            "",
            "Use this section to show downstream scientific workflow value: surrogate field prediction enables rapid "
            "screening of hydrofoil design candidates under cavitation-risk constraints, while expensive OpenFOAM is "
            "reserved for final validation of the top-ranked designs.",
        ]
    )
    path.write_text("\n".join(lines))


def plot_top_geometries(path: Path, candidates: list[Candidate], top_rows: list[dict]) -> None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    for row in top_rows:
        candidate = by_id[row["candidate_id"]]
        ax.plot(candidate.airfoil_x, candidate.airfoil_y, label=f"{candidate.family}, AoA {candidate.AoA:g}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title("Top surrogate-screened airfoil geometries")
    ax.legend(fontsize=7, loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
