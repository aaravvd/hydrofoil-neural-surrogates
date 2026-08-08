from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_case(grid_path: Path, figures_dir: Path) -> None:
    data = np.load(grid_path, allow_pickle=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    stem = grid_path.name.replace("_grid.npz", "")
    _contour(
        data,
        data["p"],
        "Pressure (Pa)",
        figures_dir / f"{stem}_pressure.png",
        cmap="viridis",
    )
    speed = np.sqrt(data["Ux"] ** 2 + data["Uy"] ** 2)
    _contour(data, speed, "Velocity magnitude (m/s)", figures_dir / f"{stem}_velocity.png", cmap="magma")
    _contour(
        data,
        data["cavitation_indicator"],
        "Cavitation indicator",
        figures_dir / f"{stem}_cavitation.png",
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )


def _contour(data, values, title: str, path: Path, cmap: str, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    mesh = ax.pcolormesh(data["grid_x"], data["grid_y"], values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(data["airfoil_x"], data["airfoil_y"], color="white", linewidth=1.0)
    ax.fill(data["airfoil_x"], data["airfoil_y"], color="black", alpha=0.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)
