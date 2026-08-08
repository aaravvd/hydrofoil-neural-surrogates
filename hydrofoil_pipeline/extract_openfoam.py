from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from hydrofoil_pipeline.cases import Case
from hydrofoil_pipeline.naca import coordinates


def extract_case(case: Case, case_dir: Path, output_path: Path) -> None:
    time_dir = latest_time_dir(case_dir)
    C = read_internal_field(time_dir / "C")
    U = read_internal_field(time_dir / "U")
    p_kinematic = read_internal_field(time_dir / "p")
    nut = read_internal_field(time_dir / "nut")
    k = read_internal_field(time_dir / "k")
    omega = read_internal_field(time_dir / "omega")

    x = C[:, 0]
    y = C[:, 1]
    Ux = U[:, 0]
    Uy = U[:, 1]
    p = case.p_inf + case.rho * p_kinematic

    Rxx = 2.0 / 3.0 * k
    Rxy = np.zeros_like(k)
    Ryy = 2.0 / 3.0 * k
    Cp = (p - case.p_inf) / (0.5 * case.rho * case.U_inf**2 + 1e-12)
    cavitation_margin = p - case.p_vap
    cavitation_indicator = (p < case.p_vap).astype(np.uint8)
    airfoil_x, airfoil_y = coordinates(case.naca, n=241, chord=case.chord)
    force_coefficients = read_force_coefficients(case_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        Ux=Ux,
        Uy=Uy,
        p=p,
        nut=nut,
        k=k,
        omega=omega,
        Rxx=Rxx,
        Rxy=Rxy,
        Ryy=Ryy,
        Cp=Cp,
        cavitation_margin=cavitation_margin,
        cavitation_indicator=cavitation_indicator,
        airfoil_x=airfoil_x,
        airfoil_y=airfoil_y,
        Re=case.Re,
        AoA=case.AoA,
        U_inf=case.U_inf,
        rho=case.rho,
        nu=case.nu,
        p_inf=case.p_inf,
        p_vap=case.p_vap,
        naca=case.naca,
        source="openfoam_simpleFoam",
        **force_coefficients,
    )


def read_force_coefficients(case_dir: Path) -> dict[str, float]:
    paths = sorted((case_dir / "postProcessing" / "forceCoeffs").glob("*/forceCoeffs.dat"))
    if not paths:
        return {}
    values = np.loadtxt(paths[-1], comments="#")
    final = values[-1] if values.ndim == 2 else values
    if final.size < 4:
        return {}
    return {
        "Cm_openfoam": float(final[1]),
        "Cd_openfoam": float(final[2]),
        "Cl_openfoam": float(final[3]),
    }


def latest_time_dir(case_dir: Path) -> Path:
    candidates = []
    for path in case_dir.iterdir():
        if path.is_dir():
            try:
                candidates.append((float(path.name), path))
            except ValueError:
                pass
    if not candidates:
        raise RuntimeError(f"No OpenFOAM time directories found in {case_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def read_internal_field(path: Path) -> np.ndarray:
    text = path.read_text()
    uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if uniform:
        token = uniform.group(1).strip()
        if token.startswith("("):
            return np.array([[float(v) for v in token.strip("()").split()]])
        return np.array([float(token)])

    match = re.search(
        r"internalField\s+nonuniform\s+List<(?P<kind>scalar|vector)>\s+(?P<n>\d+)\s*\(\s*(?P<body>.*?)\s*\)\s*;",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"Could not parse internalField in {path}")

    kind = match.group("kind")
    expected = int(match.group("n"))
    body = match.group("body")
    if kind == "scalar":
        values = np.fromstring(body, sep=" ")
        if values.size != expected:
            raise RuntimeError(f"Expected {expected} scalar values in {path}, got {values.size}")
        return values

    rows = re.findall(r"\(([^()]+)\)", body)
    values = np.array([[float(v) for v in row.split()] for row in rows])
    if values.shape != (expected, 3):
        raise RuntimeError(f"Expected {(expected, 3)} vector values in {path}, got {values.shape}")
    return values
