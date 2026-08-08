from __future__ import annotations

from pathlib import Path

import numpy as np

from hydrofoil_pipeline.cases import Case
from hydrofoil_pipeline.naca import coordinates, signed_distance_mask


def generate_case(case: Case, output_path: Path, n_points: int = 9000) -> None:
    """Generate deterministic smoke-test fields with hydrofoil-like structure."""
    rng = np.random.default_rng(abs(hash(case.id)) % (2**32))
    airfoil_x, airfoil_y = coordinates(case.naca, n=241, chord=case.chord)

    x = rng.uniform(-1.0, 2.0, n_points)
    y = rng.uniform(-0.75, 0.75, n_points)
    fluid = signed_distance_mask(x, y, airfoil_x, airfoil_y)
    x = x[fluid]
    y = y[fluid]

    alpha = np.deg2rad(case.AoA)
    U = case.U_inf
    camber_digit = int(case.naca[0])
    thickness = int(case.naca[2:]) / 100.0

    dx = x - 0.28
    dy = y - 0.02 * camber_digit
    r2 = dx**2 + dy**2 + (0.07 + 0.4 * thickness) ** 2
    circulation = 2.0 * np.pi * U * (alpha + 0.015 * camber_digit)
    vortex = circulation / (2.0 * np.pi * r2)

    blockage = np.exp(-((x - 0.45) ** 2 / 0.18 + y**2 / (0.006 + thickness**2)))
    wake = np.exp(-np.maximum(x - 0.8, 0.0) / 0.55) * np.exp(-(y / 0.12) ** 2)
    wake *= x > 0.75

    Ux = U * np.cos(alpha) - vortex * dy - 0.35 * U * wake - 0.12 * U * blockage
    Uy = U * np.sin(alpha) + vortex * dx + 0.04 * U * np.sin(2.0 * np.pi * x) * blockage
    speed2 = Ux**2 + Uy**2

    suction = (1.5 + 9.0 * abs(alpha) + 2.0 * camber_digit / 10.0) * np.exp(
        -((x - 0.18) ** 2 / 0.04 + (y - 0.04) ** 2 / 0.01)
    )
    p = case.p_inf + 0.5 * case.rho * (U**2 - speed2) - 0.5 * case.rho * U**2 * suction

    strain_scale = np.sqrt(speed2) / case.chord
    k = 0.012 * speed2 * (1.0 + 5.0 * wake + 2.0 * blockage)
    omega = np.maximum(strain_scale / (0.09 + 2.0 * thickness), 1e-6)
    nut = np.maximum(k / np.maximum(omega, 1e-9), 0.0)

    dUx_dx = np.gradient(Ux, x, edge_order=1)
    dUy_dy = np.gradient(Uy, y, edge_order=1)
    dUx_dy = np.gradient(Ux, y, edge_order=1)
    dUy_dx = np.gradient(Uy, x, edge_order=1)
    Sxx = dUx_dx
    Syy = dUy_dy
    Sxy = 0.5 * (dUx_dy + dUy_dx)
    Rxx = 2.0 / 3.0 * k - 2.0 * nut * Sxx
    Rxy = -2.0 * nut * Sxy
    Ryy = 2.0 / 3.0 * k - 2.0 * nut * Syy

    Cp = (p - case.p_inf) / (0.5 * case.rho * U**2 + 1e-12)
    cavitation_margin = p - case.p_vap
    cavitation_indicator = (p < case.p_vap).astype(np.uint8)

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
        U_inf=U,
        rho=case.rho,
        nu=case.nu,
        p_inf=case.p_inf,
        p_vap=case.p_vap,
        naca=case.naca,
        source="analytic_smoke",
    )
