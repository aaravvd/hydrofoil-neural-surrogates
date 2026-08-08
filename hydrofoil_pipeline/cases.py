from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Case:
    id: str
    naca: str
    AoA: float
    Re: float
    rho: float
    nu: float
    p_inf: float
    p_vap: float
    chord: float

    @property
    def U_inf(self) -> float:
        return self.Re * self.nu / self.chord


def load_config(path: Path) -> tuple[list[Case], dict[str, Any]]:
    cfg = yaml.safe_load(path.read_text())
    defaults = {
        "rho": float(cfg.get("rho", 997.0)),
        "nu": float(cfg.get("nu", 1.0e-6)),
        "p_inf": float(cfg.get("p_inf", 101325.0)),
        "p_vap": float(cfg.get("p_vap", 2300.0)),
        "chord": float(cfg.get("chord", 1.0)),
    }
    cases = [
        Case(
            id=str(item["id"]),
            naca=str(item["naca"]),
            AoA=float(item["AoA"]),
            Re=float(item["Re"]),
            **defaults,
        )
        for item in cfg["cases"]
    ]
    return cases, cfg


def write_metadata(cases: list[Case], path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "naca",
                "AoA",
                "Re",
                "U_inf",
                "rho",
                "nu",
                "p_inf",
                "p_vap",
                "source",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "id": case.id,
                    "naca": case.naca,
                    "AoA": case.AoA,
                    "Re": case.Re,
                    "U_inf": case.U_inf,
                    "rho": case.rho,
                    "nu": case.nu,
                    "p_inf": case.p_inf,
                    "p_vap": case.p_vap,
                    "source": source,
                }
            )
