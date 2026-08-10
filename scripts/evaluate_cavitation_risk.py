#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_model_results import load_checkpoints, predict_case, require_torch
from models.datasets import filter_force_outliers, split_paths, split_paths_by_manifest, split_paths_by_naca


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate incipient cavitation risk from surrogate Cp fields.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", default="unet,pinn,fno,deeponet")
    parser.add_argument("--ambient-pressures", default="2500,3000,5000,10000,25000,101325")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_results" / "cavitation_risk")
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-abs-force-coefficient", type=float, default=5.0)
    parser.add_argument("--split", choices=["test", "validation", "train", "all"], default="test")
    parser.add_argument("--split-strategy", choices=["random", "naca"], default="naca")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs" / "family_split.json")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch = require_torch()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    bundles = load_checkpoints(torch, args.run_dir, selected, device)
    pressures = [float(item) for item in args.ambient_pressures.split(",") if item.strip()]
    rows = []
    for model_name, bundle in bundles.items():
        if "Cp" not in bundle["norm"].target_fields:
            print(f"[skip] {model_name}: checkpoint does not predict Cp")
            continue
        cp_index = bundle["norm"].target_fields.index("Cp")
        paths, _ = filter_force_outliers(sorted(args.data_dir.glob("case_*_grid.npz")), args.max_abs_force_coefficient)
        if args.split_manifest:
            train_paths, val_paths, test_paths = split_paths_by_manifest(paths, args.split_manifest)
        else:
            splitter = split_paths_by_naca if args.split_strategy == "naca" else split_paths
            train_paths, val_paths = splitter(paths, args.val_fraction, args.seed)
            test_paths = []
        paths = test_paths if args.split == "test" else val_paths if args.split == "validation" else train_paths if args.split == "train" else paths
        for path in paths:
            truth, pred, mask, meta = predict_case(torch, bundle, path, args.batch_size, device)
            data = np.load(path, allow_pickle=True)
            q = 0.5 * float(data["rho"]) * float(data["U_inf"]) ** 2
            p_vap = float(data["p_vap"])
            for p_inf in pressures:
                truth_margin = p_inf + truth[cp_index] * q - p_vap
                pred_margin = p_inf + pred[cp_index] * q - p_vap
                valid = mask & np.isfinite(truth_margin) & np.isfinite(pred_margin)
                truth_risk = truth_margin[valid] < 0.0
                pred_risk = pred_margin[valid] < 0.0
                tp = int(np.sum(truth_risk & pred_risk))
                fp = int(np.sum(~truth_risk & pred_risk))
                fn = int(np.sum(truth_risk & ~pred_risk))
                tn = int(np.sum(~truth_risk & ~pred_risk))
                rows.append(
                    {
                        "model": model_name,
                        **meta,
                        "ambient_pressure_Pa": p_inf,
                        "truth_risky_fraction": float(np.mean(truth_risk)),
                        "pred_risky_fraction": float(np.mean(pred_risk)),
                        "min_truth_margin_Pa": float(np.min(truth_margin[valid])),
                        "min_pred_margin_Pa": float(np.min(pred_margin[valid])),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                    }
                )

    if not rows:
        raise SystemExit("No cavitation-risk rows generated.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cavitation_risk_cases.csv", rows)
    summary = summarize(rows)
    (args.output_dir / "cavitation_risk_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summarize(rows: list[dict]) -> dict:
    result = {"interpretation": "Incipient pressure-threshold risk from single-phase RANS Cp; not multiphase cavitation simulation."}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        tp = sum(row["tp"] for row in subset)
        fp = sum(row["fp"] for row in subset)
        fn = sum(row["fn"] for row in subset)
        tn = sum(row["tn"] for row in subset)
        result[model] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
            "cases_with_true_risk": sum(row["truth_risky_fraction"] > 0 for row in subset),
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
