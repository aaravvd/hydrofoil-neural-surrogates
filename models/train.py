#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from models.datasets import HydrofoilGridStore, field_names, filter_force_outliers, load_grid_paths, split_paths, split_paths_by_manifest, split_paths_by_naca


def make_split(paths, args):
    if args.split_manifest:
        train, validation, _ = split_paths_by_manifest(paths, args.split_manifest)
        return train, validation
    if args.split_strategy == "naca":
        return split_paths_by_naca(paths, args.val_fraction, args.seed)
    return split_paths(paths, args.val_fraction, args.seed)


def load_training_paths(args):
    paths = load_grid_paths(args.data_dir, args.limit_cases)
    paths, rejected = filter_force_outliers(paths, args.max_abs_force_coefficient)
    if rejected:
        print(f"[quality] excluded {len(rejected)} force outliers: {', '.join(path.stem for path in rejected)}")
    return paths


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install dependencies with:\n"
            "  python3 -m pip install -r requirements-ml.txt\n"
            "Then rerun this command."
        ) from exc
    return torch, DataLoader, TensorDataset


def masked_loss(torch, pred, target, mask, norm, bce_weight: float):
    cont = torch.as_tensor(norm.continuous_indices, dtype=torch.long, device=pred.device)
    binary = torch.as_tensor(norm.binary_indices, dtype=torch.long, device=pred.device)
    losses = []
    if cont.numel():
        err = (pred[:, cont] - target[:, cont]) ** 2
        losses.append((err * mask).sum() / mask.sum().clamp_min(1.0))
    if binary.numel():
        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, binary], target[:, binary], reduction="none")
        losses.append(bce_weight * (bce * mask).sum() / mask.sum().clamp_min(1.0))
    return sum(losses)


def pinn_residual_loss(
    torch,
    model,
    xb,
    norm,
    target_fields: list[str],
    nu_index: int,
    rho_index: int,
    weight: float,
):
    """Physics regularization for the pointwise supervised MLP.

    Evaluates steady two-dimensional continuity and inviscid momentum
    residuals using dimensional outputs and dimensional coordinates:

        r_c = du/dx + dv/dy
        r_x = u du/dx + v du/dy + (1/rho) dp/dx
        r_y = u dv/dx + v dv/dy + (1/rho) dp/dy

    Returns lambda_phys * mean(r_c^2 + r_x^2 + r_y^2).
    """
    if weight <= 0:
        return torch.zeros((), device=xb.device)

    xb_req = xb.detach().clone().requires_grad_(True)
    pred = model(xb_req)

    names = {name: i for i, name in enumerate(target_fields)}
    required = {"Ux", "Uy", "p"}
    if not required.issubset(names):
        missing = sorted(required - set(names))
        raise ValueError(
            f"Physics loss requires Ux, Uy, and p; missing {missing}"
        )

    # Convert normalized predictions back to dimensional quantities.
    out_std = torch.as_tensor(
        norm.output_std,
        dtype=pred.dtype,
        device=pred.device,
    )
    out_mean = torch.as_tensor(
        norm.output_mean,
        dtype=pred.dtype,
        device=pred.device,
    )

    u = (
        pred[:, names["Ux"]] * out_std[names["Ux"]]
        + out_mean[names["Ux"]]
    )
    v = (
        pred[:, names["Uy"]] * out_std[names["Uy"]]
        + out_mean[names["Uy"]]
    )
    p = (
        pred[:, names["p"]] * out_std[names["p"]]
        + out_mean[names["p"]]
    )

    # Input coordinates are normalized, so apply the chain rule to obtain
    # derivatives with respect to dimensional x and y.
    in_std = torch.as_tensor(
        norm.input_std,
        dtype=pred.dtype,
        device=pred.device,
    )
    in_mean = torch.as_tensor(
        norm.input_mean,
        dtype=pred.dtype,
        device=pred.device,
    )

    grad_u = torch.autograd.grad(
        u.sum(),
        xb_req,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_v = torch.autograd.grad(
        v.sum(),
        xb_req,
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_p = torch.autograd.grad(
        p.sum(),
        xb_req,
        create_graph=True,
    )[0]

    du_dx = grad_u[:, 0] / in_std[0]
    du_dy = grad_u[:, 1] / in_std[1]
    dv_dx = grad_v[:, 0] / in_std[0]
    dv_dy = grad_v[:, 1] / in_std[1]
    dp_dx = grad_p[:, 0] / in_std[0]
    dp_dy = grad_p[:, 1] / in_std[1]

    rho = xb_req[:, rho_index] * in_std[rho_index] + in_mean[rho_index]
    rho = rho.clamp_min(1e-9)

    continuity = du_dx + dv_dy
    momentum_x = u * du_dx + v * du_dy + dp_dx / rho
    momentum_y = u * dv_dx + v * dv_dy + dp_dy / rho

    physics_loss = (
        continuity.square().mean()
        + momentum_x.square().mean()
        + momentum_y.square().mean()
    )

    return weight * physics_loss


def train_point_model(args, model_name: str):
    torch, DataLoader, TensorDataset = require_torch()
    paths = load_training_paths(args)
    train_paths, val_paths = make_split(paths, args)
    targets = field_names(args.targets)
    train_store = HydrofoilGridStore(train_paths, targets)
    val_store = HydrofoilGridStore(val_paths or train_paths, targets)
    val_store.normalization = train_store.normalization

    x_train, y_train = train_store.point_arrays(args.max_points_per_case, args.seed)
    x_val, y_val = val_store.point_arrays(args.max_points_per_case, args.seed + 1)

    from models.architectures import build_model

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, x_train.shape[1], y_train.shape[1], args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=args.batch_size)

    best = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()
    nu_index = train_store.normalization.input_fields.index("nu")
    rho_index = train_store.normalization.input_fields.index("rho")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mask = torch.ones(pred.shape[0], 1, device=device)
            loss = masked_loss(torch, pred, yb, mask, train_store.normalization, args.bce_weight)
            if model_name == "pinn":
                loss = loss + pinn_residual_loss(
    torch,
    model,
    xb,
    train_store.normalization,
    targets,
    rho_index,
    args.physics_weight,
)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= len(train_loader.dataset)
        val_loss = evaluate_point(torch, model, val_loader, train_store.normalization, device, args.bce_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            save_checkpoint(torch, args.output_dir, model_name, model, train_store.normalization, args, history)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"{model_name} epoch={epoch:04d} train={train_loss:.6g} val={val_loss:.6g}")
        if args.early_stopping_patience and epoch - best_epoch >= args.early_stopping_patience:
            print(f"{model_name} early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    write_metrics(args.output_dir, model_name, history, best, best_epoch, len(train_paths), len(val_paths), model, time.perf_counter() - started, args)


def evaluate_point(torch, model, loader, norm, device, bce_weight):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            mask = torch.ones(pred.shape[0], 1, device=device)
            loss = masked_loss(torch, pred, yb, mask, norm, bce_weight)
            total += loss.item() * xb.shape[0]
    return total / len(loader.dataset)


def train_grid_model(args, model_name: str):
    torch, DataLoader, TensorDataset = require_torch()
    paths = load_training_paths(args)
    train_paths, val_paths = make_split(paths, args)
    targets = field_names(args.targets)
    train_store = HydrofoilGridStore(train_paths, targets)
    val_store = HydrofoilGridStore(val_paths or train_paths, targets)
    val_store.normalization = train_store.normalization
    x_train, y_train, m_train = train_store.operator_arrays()
    x_val, y_val, m_val = val_store.operator_arrays()

    from models.architectures import build_model

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, x_train.shape[1], y_train.shape[1], args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(m_train)), batch_size=args.operator_batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(m_val)), batch_size=args.operator_batch_size)
    best = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            pred = model(xb)
            loss = masked_loss(torch, pred, yb, mb, train_store.normalization, args.bce_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.shape[0]
        train_loss /= len(train_loader.dataset)
        val_loss = evaluate_fno(torch, model, val_loader, train_store.normalization, device, args.bce_weight)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            save_checkpoint(torch, args.output_dir, model_name, model, train_store.normalization, args, history)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"{model_name} epoch={epoch:04d} train={train_loss:.6g} val={val_loss:.6g}")
        if args.early_stopping_patience and epoch - best_epoch >= args.early_stopping_patience:
            print(f"{model_name} early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    write_metrics(args.output_dir, model_name, history, best, best_epoch, len(train_paths), len(val_paths), model, time.perf_counter() - started, args)


def evaluate_fno(torch, model, loader, norm, device, bce_weight):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb, mb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            pred = model(xb)
            loss = masked_loss(torch, pred, yb, mb, norm, bce_weight)
            total += loss.item() * xb.shape[0]
    return total / len(loader.dataset)


def save_checkpoint(torch, output_dir: Path, model_name: str, model, norm, args, history):
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": model_name,
            "target_fields": norm.target_fields,
            "input_fields": norm.input_fields,
            "args": vars(args),
            "history": history,
        },
        run_dir / "best.pt",
    )
    norm.save(run_dir / "normalization.npz")


def write_metrics(output_dir: Path, model_name: str, history: list[dict], best: float, best_epoch: int, n_train: int, n_val: int, model, elapsed: float, args):
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "n_train_cases": n_train,
        "n_val_cases": n_val,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": elapsed,
        "split_strategy": args.split_strategy,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "seed": args.seed,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hydrofoil CNN-U-Net, PINN, FNO, and DeepONet surrogates.")
    parser.add_argument("--model", choices=["dnn", "unet", "cnn_unet", "pinn", "fno", "deeponet", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed_grids")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training_runs")
    parser.add_argument("--targets", default=",".join(field_names(None)))
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--max-abs-force-coefficient", type=float, default=5.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-strategy", choices=["random", "naca"], default="naca")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs" / "family_split.json")
    parser.add_argument("--max-points-per-case", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--operator-batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--basis", type=int, default=128)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight", type=float, default=0.2)
    parser.add_argument("--physics-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch, _, _ = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected = ["unet", "pinn", "fno", "deeponet"] if args.model == "all" else [args.model]
    for name in selected:
        if name == "cnn_unet":
            name = "unet"
        if name in {"fno", "unet"}:
            train_grid_model(args, name)
        else:
            train_point_model(args, name)


if __name__ == "__main__":
    main()
