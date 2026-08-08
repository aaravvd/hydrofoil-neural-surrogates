from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

from hydrofoil_pipeline.naca import geometry_descriptors, signed_distance, signed_distance_mask


FIELD_NAMES = ["Ux", "Uy", "p", "nut", "k", "omega", "Rxx", "Rxy", "Ryy", "Cp", "cavitation_margin"]


def grid_case(raw_path: Path, output_path: Path, grid_cfg: dict) -> None:
    data = np.load(raw_path, allow_pickle=True)
    gx = np.linspace(float(grid_cfg["x_min"]), float(grid_cfg["x_max"]), int(grid_cfg["nx"]))
    gy = np.linspace(float(grid_cfg["y_min"]), float(grid_cfg["y_max"]), int(grid_cfg["ny"]))
    X, Y = np.meshgrid(gx, gy)
    pts = np.column_stack([data["x"], data["y"]])

    gridded = {"grid_x": X, "grid_y": Y}
    fluid_mask = signed_distance_mask(X, Y, data["airfoil_x"], data["airfoil_y"])
    gridded["sdf"] = signed_distance(X, Y, data["airfoil_x"], data["airfoil_y"]).astype(np.float32)
    for name in FIELD_NAMES:
        linear = griddata(pts, data[name], (X, Y), method="linear")
        nearest = griddata(pts, data[name], (X, Y), method="nearest")
        values = np.where(np.isfinite(linear), linear, nearest)
        gridded[name] = np.where(fluid_mask, values, np.nan)

    p_abs = gridded["p"]
    gridded["cavitation_indicator"] = ((p_abs < float(data["p_vap"])) & fluid_mask).astype(np.uint8)
    gridded["fluid_mask"] = fluid_mask.astype(np.uint8)
    if "Cl_openfoam" in data.files and "Cd_openfoam" in data.files:
        gridded["Cl"] = np.where(fluid_mask, float(data["Cl_openfoam"]), np.nan).astype(np.float32)
        gridded["Cd"] = np.where(fluid_mask, float(data["Cd_openfoam"]), np.nan).astype(np.float32)
    for scalar in ["Re", "AoA", "U_inf", "rho", "nu", "p_inf", "p_vap"]:
        gridded[scalar] = data[scalar]
    gridded["airfoil_x"] = data["airfoil_x"]
    gridded["airfoil_y"] = data["airfoil_y"]
    gridded["naca"] = data["naca"]
    gridded["source"] = data["source"]
    max_camber, camber_position, thickness_ratio = geometry_descriptors(data["airfoil_x"], data["airfoil_y"])
    gridded["max_camber"] = max_camber
    gridded["camber_position"] = camber_position
    gridded["thickness_ratio"] = thickness_ratio
    for scalar in ["Cm_openfoam", "Cd_openfoam", "Cl_openfoam"]:
        if scalar in data.files:
            gridded[scalar] = data[scalar]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **gridded)
