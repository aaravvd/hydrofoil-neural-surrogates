from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Naca4:
    code: str
    m: float
    p: float
    t: float


def parse_naca4(code: str) -> Naca4:
    clean = code.upper().replace("NACA", "").strip()
    if len(clean) != 4 or not clean.isdigit():
        raise ValueError(f"Expected a 4-digit NACA code, got {code!r}")
    return Naca4(
        code=clean,
        m=int(clean[0]) / 100.0,
        p=int(clean[1]) / 10.0,
        t=int(clean[2:]) / 100.0,
    )


def coordinates(code: str, n: int = 201, chord: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return closed upper/lower NACA 4-digit coordinates."""
    foil = parse_naca4(code)
    return coordinates_from_parameters(foil.m, foil.p, foil.t, n=n, chord=chord)


def coordinates_from_parameters(
    m: float, p: float, t: float, n: int = 201, chord: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a closed NACA-like four-digit geometry from continuous parameters."""
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 1e-12))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if m > 0.0 and p > 0.0:
        fore = x < p
        aft = ~fore
        yc[fore] = m / p**2 * (2.0 * p * x[fore] - x[fore] ** 2)
        dyc_dx[fore] = 2.0 * m / p**2 * (p - x[fore])
        yc[aft] = m / (1.0 - p) ** 2 * (
            (1.0 - 2.0 * p) + 2.0 * p * x[aft] - x[aft] ** 2
        )
        dyc_dx[aft] = 2.0 * m / (1.0 - p) ** 2 * (p - x[aft])

    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_closed = np.concatenate([xu[::-1], xl[1:]]) * chord
    y_closed = np.concatenate([yu[::-1], yl[1:]]) * chord
    return x_closed, y_closed


def signed_distance_mask(
    x: np.ndarray,
    y: np.ndarray,
    airfoil_x: np.ndarray,
    airfoil_y: np.ndarray,
) -> np.ndarray:
    """Approximate point-in-polygon mask: True for fluid, False inside foil."""
    px = x.ravel()
    py = y.ravel()
    ax = airfoil_x
    ay = airfoil_y
    inside = np.zeros_like(px, dtype=bool)
    j = len(ax) - 1
    for i in range(len(ax)):
        intersects = ((ay[i] > py) != (ay[j] > py)) & (
            px < (ax[j] - ax[i]) * (py - ay[i]) / (ay[j] - ay[i] + 1e-15) + ax[i]
        )
        inside ^= intersects
        j = i
    return (~inside).reshape(x.shape)


def signed_distance(
    x: np.ndarray,
    y: np.ndarray,
    airfoil_x: np.ndarray,
    airfoil_y: np.ndarray,
) -> np.ndarray:
    """Return positive distance in fluid and negative distance inside the foil."""
    fluid = signed_distance_mask(x, y, airfoil_x, airfoil_y)
    px = x[..., None]
    py = y[..., None]
    ax = np.asarray(airfoil_x)
    ay = np.asarray(airfoil_y)
    if ax[0] != ax[-1] or ay[0] != ay[-1]:
        ax = np.r_[ax, ax[0]]
        ay = np.r_[ay, ay[0]]
    x0, y0 = ax[:-1], ay[:-1]
    dx, dy = ax[1:] - x0, ay[1:] - y0
    denom = dx * dx + dy * dy + 1e-15
    projection = np.clip(((px - x0) * dx + (py - y0) * dy) / denom, 0.0, 1.0)
    qx = x0 + projection * dx
    qy = y0 + projection * dy
    distance = np.sqrt(np.min((px - qx) ** 2 + (py - qy) ** 2, axis=-1))
    return np.where(fluid, distance, -distance)


def geometry_descriptors(airfoil_x: np.ndarray, airfoil_y: np.ndarray, chord: float = 1.0) -> tuple[float, float, float]:
    """Estimate maximum camber, camber location, and thickness ratio from a closed profile."""
    ax = np.asarray(airfoil_x, dtype=float) / chord
    ay = np.asarray(airfoil_y, dtype=float) / chord
    leading_edge = int(np.argmin(ax))
    branch_a = (ax[: leading_edge + 1][::-1], ay[: leading_edge + 1][::-1])
    branch_b = (ax[leading_edge:], ay[leading_edge:])
    sample_x = np.linspace(max(branch_a[0].min(), branch_b[0].min()), min(branch_a[0].max(), branch_b[0].max()), 401)
    ya = np.interp(sample_x, branch_a[0], branch_a[1])
    yb = np.interp(sample_x, branch_b[0], branch_b[1])
    upper = np.maximum(ya, yb)
    lower = np.minimum(ya, yb)
    camber = 0.5 * (upper + lower)
    thickness = upper - lower
    index = int(np.argmax(np.abs(camber)))
    if abs(camber[index]) < 1e-5:
        return 0.0, 0.5, float(np.max(thickness))
    return float(camber[index]), float(sample_x[index]), float(np.max(thickness))
