#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / ".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from models.architectures import build_model
from models.datasets import HydrofoilGridStore, denormalize_output_grid, load_grid_paths, load_normalization, split_paths


MODEL_ORDER = ["unet", "pinn", "fno", "deeponet"]


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for model visualization. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-ml.txt"
        ) from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot truth vs model predictions on validation hydrofoil cases.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs", help="Directory containing dnn/pinn/fno/deeponet subfolders.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed_grids")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "model_predictions")
    parser.add_argument("--field", default="p", help="Target field to visualize, e.g. p, Cp, cavitation_margin.")
    parser.add_argument("--models", default="all", help="Comma list from dnn,pinn,fno,deeponet or all.")
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--case", action="append", default=[], help="Specific case id or grid path. Can be repeated.")
    parser.add_argument("--split", choices=["validation", "train", "all"], default="validation")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = MODEL_ORDER if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]

    checkpoints = load_checkpoints(torch, args.run_dir, selected, device)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {args.run_dir}. Expected subfolders like dnn/best.pt.")

    first_norm = next(iter(checkpoints.values()))["norm"]
    if args.field not in first_norm.target_fields:
        raise SystemExit(f"Field {args.field!r} is not in checkpoint targets: {first_norm.target_fields}")

    case_paths = choose_cases(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in case_paths:
        predictions = {}
        truth, mask, plot_data = load_truth(path, args.field)
        for model_name, bundle in checkpoints.items():
            predictions[model_name] = predict_case(torch, bundle, path, args.field, args.batch_size, device)
        out_path = args.output_dir / f"{path.name.replace('_grid.npz', '')}_{args.field}_predictions.png"
        plot_predictions(plot_data, args.field, truth, mask, predictions, out_path)
        print(f"[ok] wrote {out_path}")


def load_checkpoints(torch, run_dir: Path, selected: list[str], device) -> dict[str, dict]:
    bundles = {}
    for model_name in selected:
        ckpt_path = run_dir / model_name / "best.pt"
        norm_path = run_dir / model_name / "normalization.npz"
        if not ckpt_path.exists() or not norm_path.exists():
            print(f"[skip] {model_name}: missing {ckpt_path} or {norm_path}")
            continue
        ckpt = load_local_checkpoint(torch, ckpt_path, device)
        norm = load_normalization(norm_path)
        train_args = SimpleNamespace(**ckpt.get("args", {}))
        ensure_model_defaults(train_args)
        model = build_model(model_name, len(norm.input_fields), len(norm.target_fields), train_args).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        bundles[model_name] = {"model": model, "norm": norm}
    return bundles


def load_local_checkpoint(torch, path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def ensure_model_defaults(args: SimpleNamespace) -> None:
    defaults = {"width": 128, "depth": 4, "basis": 128, "modes": 16}
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def choose_cases(args) -> list[Path]:
    if args.case:
        paths = []
        for item in args.case:
            candidate = Path(item)
            if candidate.exists():
                paths.append(candidate)
                continue
            by_id = args.data_dir / f"{item.replace('_grid.npz', '')}_grid.npz"
            if not by_id.exists():
                raise FileNotFoundError(f"Could not find case {item!r} at {by_id}")
            paths.append(by_id)
        return paths[: args.num_cases]

    paths = load_grid_paths(args.data_dir, args.limit_cases)
    train_paths, val_paths = split_paths(paths, args.val_fraction, args.seed)
    if args.split == "validation":
        chosen = val_paths or paths
    elif args.split == "train":
        chosen = train_paths
    else:
        chosen = paths
    return chosen[: args.num_cases]


def load_truth(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    mask = data["fluid_mask"].astype(bool)
    values = data[field].astype(np.float32)
    truth = np.where(mask, values, np.nan)
    return truth, mask, {key: data[key] for key in ["grid_x", "grid_y", "airfoil_x", "airfoil_y"]}


def predict_case(torch, bundle: dict, path: Path, field: str, batch_size: int, device) -> np.ndarray:
    norm = bundle["norm"]
    store = HydrofoilGridStore([path], norm.target_fields)
    store.normalization = norm
    x_grid, _, mask_grid = store.operator_arrays()
    model = bundle["model"]
    field_idx = norm.target_fields.index(field)

    with torch.no_grad():
        if model.__class__.__name__ in {"FNO2d", "UNet2d"}:
            xb = torch.from_numpy(x_grid).to(device)
            pred = model(xb).cpu().numpy()[0]
        else:
            channels, height, width = x_grid.shape[1:]
            flat = np.moveaxis(x_grid[0], 0, -1).reshape(height * width, channels)
            chunks = []
            for start in range(0, flat.shape[0], batch_size):
                xb = torch.from_numpy(flat[start : start + batch_size]).to(device)
                chunks.append(model(xb).cpu().numpy())
            pred_flat = np.concatenate(chunks, axis=0)
            pred = np.moveaxis(pred_flat.reshape(height, width, -1), -1, 0)

    pred = denormalize_output_grid(pred, norm)
    if field_idx in set(norm.binary_indices.tolist()):
        pred[field_idx] = 1.0 / (1.0 + np.exp(-np.clip(pred[field_idx], -60.0, 60.0)))
    mask = mask_grid[0, 0].astype(bool)
    return np.where(mask, pred[field_idx], np.nan)


def plot_predictions(plot_data: dict, field: str, truth: np.ndarray, mask: np.ndarray, predictions: dict[str, np.ndarray], out_path: Path) -> None:
    names = ["truth", *predictions.keys()]
    values = [truth, *predictions.values()]
    finite = [arr[np.isfinite(arr)] for arr in values if np.isfinite(arr).any()]
    finite_values = np.concatenate(finite) if finite else np.array([0.0, 1.0])
    vmin, vmax = np.percentile(finite_values, [1, 99])
    if math.isclose(float(vmin), float(vmax)):
        vmin, vmax = float(vmin) - 1.0, float(vmax) + 1.0

    ncols = len(names)
    fig, axes = plt.subplots(2, ncols, figsize=(4.0 * ncols, 6.2), constrained_layout=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (name, arr) in enumerate(zip(names, values)):
        draw_field(axes[0, col], plot_data, arr, f"{name} {field}", "viridis", vmin, vmax)
        if name == "truth":
            axes[1, col].axis("off")
            axes[1, col].set_title("absolute error")
        else:
            err = np.where(mask, np.abs(arr - truth), np.nan)
            err_max = np.nanpercentile(err, 99) if np.isfinite(err).any() else 1.0
            draw_field(axes[1, col], plot_data, err, f"{name} abs error", "magma", 0.0, err_max)

    fig.suptitle(out_path.name.replace("_", " "), fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def draw_field(ax, plot_data: dict, values: np.ndarray, title: str, cmap: str, vmin: float, vmax: float) -> None:
    mesh = ax.pcolormesh(plot_data["grid_x"], plot_data["grid_y"], values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(plot_data["airfoil_x"], plot_data["airfoil_y"], color="white", linewidth=0.8)
    ax.fill(plot_data["airfoil_x"], plot_data["airfoil_y"], color="black", alpha=0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, shrink=0.8)


if __name__ == "__main__":
    main()
