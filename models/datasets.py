from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_TARGET_FIELDS = [
    "Ux",
    "Uy",
    "p",
    "nut",
    "k",
    "omega",
    "Rxx",
    "Rxy",
    "Ryy",
    "Cp",
    "cavitation_margin",
    "cavitation_indicator",
    "Cl",
    "Cd",
]

CASE_PARAM_FIELDS = ["Re", "AoA", "U_inf", "rho", "nu", "p_inf", "p_vap"]
GEOMETRY_PARAM_FIELDS = ["max_camber", "camber_position", "thickness_ratio"]


@dataclass(frozen=True)
class Normalization:
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray
    continuous_indices: np.ndarray
    binary_indices: np.ndarray
    target_fields: list[str]
    input_fields: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            input_mean=self.input_mean,
            input_std=self.input_std,
            output_mean=self.output_mean,
            output_std=self.output_std,
            continuous_indices=self.continuous_indices,
            binary_indices=self.binary_indices,
            target_fields=np.array(self.target_fields),
            input_fields=np.array(self.input_fields),
        )


def load_grid_paths(data_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(data_dir.glob("case_*_grid.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No processed grids found in {data_dir}")
    return paths


def filter_force_outliers(paths: list[Path], max_abs_force_coefficient: float | None) -> tuple[list[Path], list[Path]]:
    if max_abs_force_coefficient is None:
        return paths, []
    kept, rejected = [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            if not {"Cl_openfoam", "Cd_openfoam"}.issubset(data.files):
                kept.append(path)
                continue
            values = [float(data["Cl_openfoam"]), float(data["Cd_openfoam"])]
            (kept if all(np.isfinite(value) and abs(value) <= max_abs_force_coefficient for value in values) else rejected).append(path)
    return kept, rejected


def split_paths(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(paths))
    rng.shuffle(order)
    n_val = max(1, int(round(len(paths) * val_fraction))) if len(paths) > 1 else 0
    val_idx = set(order[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    if not train:
        train, val = val, train
    return train, val


def split_paths_by_naca(paths: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Hold out complete NACA families to measure geometry generalization."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            groups.setdefault(str(data["naca"]), []).append(path)
    labels = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    n_val = max(1, int(round(len(labels) * val_fraction))) if len(labels) > 1 else 0
    val_labels = set(labels[:n_val])
    train = [path for label in labels if label not in val_labels for path in groups[label]]
    val = [path for label in labels if label in val_labels for path in groups[label]]
    if not train:
        train, val = val, train
    return sorted(train), sorted(val)


def field_names(fields: str | Iterable[str] | None) -> list[str]:
    if fields is None:
        return list(DEFAULT_TARGET_FIELDS)
    if isinstance(fields, str):
        return [f.strip() for f in fields.split(",") if f.strip()]
    return list(fields)


class HydrofoilGridStore:
    """Loads processed grids and exposes point and operator-learning arrays."""

    def __init__(self, paths: list[Path], target_fields: list[str] | None = None):
        self.paths = paths
        self.target_fields = target_fields or list(DEFAULT_TARGET_FIELDS)
        self.input_fields = ["x", "y", *CASE_PARAM_FIELDS, *GEOMETRY_PARAM_FIELDS, "fluid_mask", "sdf"]
        self._cases = [self._load_case(path) for path in paths]
        self.normalization = self._fit_normalization()

    def _load_case(self, path: Path) -> dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        missing = [name for name in self.target_fields if name not in data.files]
        if missing:
            raise KeyError(f"{path} is missing target fields: {missing}")

        gx = data["grid_x"].astype(np.float32)
        gy = data["grid_y"].astype(np.float32)
        mask = data["fluid_mask"].astype(bool)
        if "sdf" in data.files:
            sdf = data["sdf"].astype(np.float32)
        else:
            sdf = np.zeros_like(gx, dtype=np.float32)

        params = np.stack([np.asarray(data[name]).astype(np.float32) for name in CASE_PARAM_FIELDS])
        geometry_params = np.stack([np.asarray(data[name]).astype(np.float32) for name in GEOMETRY_PARAM_FIELDS])
        targets = []
        for name in self.target_fields:
            if name == "cavitation_indicator":
                if name in data.files:
                    field = data[name].astype(np.float32)
                else:
                    field = (data["p"] < float(data["p_vap"])).astype(np.float32)
            else:
                field = data[name].astype(np.float32)
            targets.append(field)

        y = np.stack(targets, axis=0)
        valid = mask & np.all(np.isfinite(y), axis=0)
        x_channels = [gx, gy]
        x_channels.extend([np.full_like(gx, p, dtype=np.float32) for p in params])
        x_channels.extend([np.full_like(gx, p, dtype=np.float32) for p in geometry_params])
        x_channels.append(mask.astype(np.float32))
        x_channels.append(sdf.astype(np.float32))
        x = np.stack(x_channels, axis=0)

        y = np.where(valid[None, ...], y, 0.0).astype(np.float32)
        x = np.where(mask[None, ...], x, 0.0).astype(np.float32)
        return {"x": x, "y": y, "mask": valid.astype(np.float32), "path": np.array(str(path))}

    def _fit_normalization(self) -> Normalization:
        x_points = []
        y_points = []
        for case in self._cases:
            valid = case["mask"].astype(bool)
            x_points.append(case["x"][:, valid].T)
            y_points.append(case["y"][:, valid].T)
        x_all = np.concatenate(x_points, axis=0)
        y_all = np.concatenate(y_points, axis=0)
        binary = np.array([i for i, name in enumerate(self.target_fields) if name == "cavitation_indicator"], dtype=np.int64)
        continuous = np.array([i for i, name in enumerate(self.target_fields) if name != "cavitation_indicator"], dtype=np.int64)
        y_mean = np.zeros(y_all.shape[1], dtype=np.float32)
        y_std = np.ones(y_all.shape[1], dtype=np.float32)
        if continuous.size:
            y_mean[continuous] = y_all[:, continuous].mean(axis=0)
            y_std[continuous] = y_all[:, continuous].std(axis=0)
        x_mean = x_all.mean(axis=0).astype(np.float32)
        x_std = x_all.std(axis=0).astype(np.float32)
        return Normalization(
            input_mean=x_mean,
            input_std=np.maximum(x_std, 1e-6),
            output_mean=y_mean,
            output_std=np.maximum(y_std, 1e-6),
            continuous_indices=continuous,
            binary_indices=binary,
            target_fields=list(self.target_fields),
            input_fields=list(self.input_fields),
        )

    def point_arrays(self, max_points_per_case: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        xs = []
        ys = []
        for case in self._cases:
            valid_idx = np.flatnonzero(case["mask"].reshape(-1) > 0.5)
            if max_points_per_case is not None and valid_idx.size > max_points_per_case:
                valid_idx = rng.choice(valid_idx, size=max_points_per_case, replace=False)
            x = case["x"].reshape(case["x"].shape[0], -1)[:, valid_idx].T
            y = case["y"].reshape(case["y"].shape[0], -1)[:, valid_idx].T
            xs.append(self.normalize_inputs(x))
            ys.append(self.normalize_outputs(y))
        return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)

    def operator_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = []
        ys = []
        masks = []
        for case in self._cases:
            x = self.normalize_input_grid(case["x"])
            y = self.normalize_output_grid(case["y"])
            xs.append(x)
            ys.append(y)
            masks.append(case["mask"][None, ...].astype(np.float32))
        return np.stack(xs), np.stack(ys), np.stack(masks)

    def normalize_inputs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.normalization.input_mean) / self.normalization.input_std

    def normalize_outputs(self, y: np.ndarray) -> np.ndarray:
        out = y.copy()
        idx = self.normalization.continuous_indices
        if idx.size:
            out[:, idx] = (out[:, idx] - self.normalization.output_mean[idx]) / self.normalization.output_std[idx]
        return out

    def normalize_input_grid(self, x: np.ndarray) -> np.ndarray:
        return (x - self.normalization.input_mean[:, None, None]) / self.normalization.input_std[:, None, None]

    def normalize_output_grid(self, y: np.ndarray) -> np.ndarray:
        out = y.copy()
        idx = self.normalization.continuous_indices
        if idx.size:
            out[idx] = (out[idx] - self.normalization.output_mean[idx, None, None]) / self.normalization.output_std[idx, None, None]
        return out


def load_normalization(path: Path) -> Normalization:
    data = np.load(path, allow_pickle=True)
    return Normalization(
        input_mean=data["input_mean"].astype(np.float32),
        input_std=data["input_std"].astype(np.float32),
        output_mean=data["output_mean"].astype(np.float32),
        output_std=data["output_std"].astype(np.float32),
        continuous_indices=data["continuous_indices"].astype(np.int64),
        binary_indices=data["binary_indices"].astype(np.int64),
        target_fields=[str(item) for item in data["target_fields"].tolist()],
        input_fields=[str(item) for item in data["input_fields"].tolist()],
    )


def denormalize_output_grid(y: np.ndarray, norm: Normalization) -> np.ndarray:
    out = y.copy()
    idx = norm.continuous_indices
    if idx.size:
        out[idx] = out[idx] * norm.output_std[idx, None, None] + norm.output_mean[idx, None, None]
    return out
