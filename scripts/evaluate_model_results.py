#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from models.architectures import build_model
from models.datasets import HydrofoilGridStore, denormalize_output_grid, filter_force_outliers, load_grid_paths, load_normalization, split_paths, split_paths_by_manifest, split_paths_by_naca


MODEL_ORDER = ["unet", "pinn", "fno", "deeponet"]


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for evaluation. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-ml.txt"
        ) from exc
    return torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained hydrofoil surrogate models on validation cases.")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed_grids")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_results")
    parser.add_argument("--models", default="all")
    parser.add_argument("--split", choices=["test", "validation", "train", "all"], default="test")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-strategy", choices=["random", "naca"], default="naca")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs" / "family_split.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--max-abs-force-coefficient", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = MODEL_ORDER if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    bundles = load_checkpoints(torch, args.run_dir, selected, device)
    if not bundles:
        raise SystemExit(f"No checkpoints found in {args.run_dir}")

    paths = choose_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    field_rows = []
    case_rows = []
    for model_name, bundle in bundles.items():
        print(f"[eval] {model_name}: {len(paths)} cases")
        sums = {field: empty_sums() for field in bundle["norm"].target_fields}
        for path in paths:
            truth, pred, mask, meta = predict_case(torch, bundle, path, args.batch_size, device)
            for field_index, field in enumerate(bundle["norm"].target_fields):
                update_sums(sums[field], truth[field_index], pred[field_index], mask)
            case_rows.append(case_summary(model_name, path, truth, pred, mask, bundle["norm"].target_fields, meta))
        for field, values in sums.items():
            field_rows.append(metric_row(model_name, field, values))

    write_csv(args.output_dir / "field_metrics.csv", field_rows)
    write_csv(args.output_dir / "case_metrics.csv", case_rows)
    write_json_summary(args.output_dir / "summary.json", args, paths, field_rows, case_rows)
    print(f"[ok] wrote {args.output_dir / 'field_metrics.csv'}")
    print(f"[ok] wrote {args.output_dir / 'case_metrics.csv'}")
    print(f"[ok] wrote {args.output_dir / 'summary.json'}")


def load_checkpoints(torch, run_dir: Path, selected: list[str], device) -> dict[str, dict]:
    bundles = {}
    for model_name in selected:
        ckpt_path = run_dir / model_name / "best.pt"
        norm_path = run_dir / model_name / "normalization.npz"
        if not ckpt_path.exists() or not norm_path.exists():
            print(f"[skip] {model_name}: missing checkpoint or normalization")
            continue
        ckpt = load_local_checkpoint(torch, ckpt_path, device)
        norm = load_normalization(norm_path)
        train_args = SimpleNamespace(**ckpt.get("args", {}))
        for key, value in {"width": 128, "depth": 4, "basis": 128, "modes": 16}.items():
            if not hasattr(train_args, key):
                setattr(train_args, key, value)
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


def choose_paths(args) -> list[Path]:
    paths = load_grid_paths(args.data_dir, args.limit_cases)
    paths, rejected = filter_force_outliers(paths, args.max_abs_force_coefficient)
    if rejected:
        print(f"[quality] excluded {len(rejected)} force outliers")
    if args.split_manifest:
        train_paths, val_paths, test_paths = split_paths_by_manifest(paths, args.split_manifest)
    else:
        splitter = split_paths_by_naca if args.split_strategy == "naca" else split_paths
        train_paths, val_paths = splitter(paths, args.val_fraction, args.seed)
        test_paths = []
    if args.split == "test":
        return test_paths
    if args.split == "validation":
        return val_paths or paths
    if args.split == "train":
        return train_paths
    return paths


def predict_case(torch, bundle: dict, path: Path, batch_size: int, device):
    norm = bundle["norm"]
    store = HydrofoilGridStore([path], norm.target_fields)
    store.normalization = norm
    x_grid, y_grid, mask_grid = store.operator_arrays()
    model = bundle["model"]
    with torch.no_grad():
        if model.__class__.__name__ in {"FNO2d", "UNet2d"}:
            pred = model(torch.from_numpy(x_grid).to(device)).cpu().numpy()[0]
        else:
            channels, height, width = x_grid.shape[1:]
            flat = np.moveaxis(x_grid[0], 0, -1).reshape(height * width, channels)
            chunks = []
            for start in range(0, flat.shape[0], batch_size):
                chunks.append(model(torch.from_numpy(flat[start : start + batch_size]).to(device)).cpu().numpy())
            pred = np.moveaxis(np.concatenate(chunks, axis=0).reshape(height, width, -1), -1, 0)
    truth = denormalize_output_grid(y_grid[0], norm)
    pred = denormalize_output_grid(pred, norm)
    for idx in norm.binary_indices:
        pred[idx] = 1.0 / (1.0 + np.exp(-np.clip(pred[idx], -60.0, 60.0)))
    mask = mask_grid[0, 0].astype(bool)
    data = np.load(path, allow_pickle=True)
    meta = {
        "case": path.name.replace("_grid.npz", ""),
        "naca": str(data["naca"]) if "naca" in data.files else "",
        "AoA": float(data["AoA"]) if "AoA" in data.files else np.nan,
        "Re": float(data["Re"]) if "Re" in data.files else np.nan,
        "U_inf": float(data["U_inf"]) if "U_inf" in data.files else np.nan,
    }
    return truth, pred, mask, meta


def empty_sums() -> dict[str, float]:
    return {"n": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_true": 0.0, "sum_true_sq": 0.0, "sum_pred": 0.0}


def update_sums(sums: dict[str, float], truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> None:
    valid = mask & np.isfinite(truth) & np.isfinite(pred)
    if not valid.any():
        return
    t = truth[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    err = p - t
    sums["n"] += float(t.size)
    sums["sum_abs"] += float(np.abs(err).sum())
    sums["sum_sq"] += float(np.square(err).sum())
    sums["sum_true"] += float(t.sum())
    sums["sum_true_sq"] += float(np.square(t).sum())
    sums["sum_pred"] += float(p.sum())


def metric_row(model: str, field: str, sums: dict[str, float]) -> dict[str, float | str]:
    n = max(sums["n"], 1.0)
    mae = sums["sum_abs"] / n
    rmse = (sums["sum_sq"] / n) ** 0.5
    mean_true = sums["sum_true"] / n
    sst = sums["sum_true_sq"] - sums["sum_true"] ** 2 / n
    r2 = 1.0 - sums["sum_sq"] / sst if sst > 1e-20 else float("nan")
    rel_rmse = rmse / (abs(mean_true) + 1e-12)
    return {"model": model, "field": field, "n": int(sums["n"]), "mae": mae, "rmse": rmse, "relative_rmse": rel_rmse, "r2": r2}


def case_summary(model: str, path: Path, truth: np.ndarray, pred: np.ndarray, mask: np.ndarray, fields: list[str], meta: dict) -> dict:
    row = {"model": model, **meta}
    for field in ["p", "Cp", "cavitation_margin", "cavitation_indicator"]:
        if field not in fields:
            continue
        idx = fields.index(field)
        valid = mask & np.isfinite(truth[idx]) & np.isfinite(pred[idx])
        if not valid.any():
            continue
        err = pred[idx][valid] - truth[idx][valid]
        row[f"{field}_rmse"] = float(np.sqrt(np.mean(np.square(err))))
        row[f"{field}_mae"] = float(np.mean(np.abs(err)))
        row[f"{field}_truth_min"] = float(np.min(truth[idx][valid]))
        row[f"{field}_pred_min"] = float(np.min(pred[idx][valid]))
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json_summary(path: Path, args, paths: list[Path], field_rows: list[dict], case_rows: list[dict]) -> None:
    payload = {
        "run_dir": str(args.run_dir),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "val_fraction": args.val_fraction,
        "split_strategy": args.split_strategy,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "seed": args.seed,
        "n_cases": len(paths),
        "cases": [p.name.replace("_grid.npz", "") for p in paths],
        "best_pressure_by_rmse": sorted([r for r in field_rows if r["field"] == "p"], key=lambda r: r["rmse"]),
        "field_metrics": field_rows,
        "case_metrics": case_rows,
    }
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
