#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


FULL_NACAS = [
    "0006",
    "0008",
    "0010",
    "0012",
    "0015",
    "0018",
    "2412",
    "2415",
    "4412",
    "4415",
    "4418",
    "4421",
]
AOAS = [-4, -2, 0, 2, 4, 6, 8, 10]
RES = [1e5, 2e5, 5e5, 1e6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["pilot", "full"], default="full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.preset == "pilot":
        nacas = ["0006", "0008", "0012", "2412", "4415"]
        aoas = [-2, 0, 4, 6, 8]
        res = [1e5, 1e5, 2e5, 5e5, 1e6]
        cases = [
            {"id": f"case_{i + 1:03d}", "naca": naca, "AoA": aoa, "Re": re}
            for i, (naca, aoa, re) in enumerate(zip(nacas, aoas, res))
        ]
    else:
        cases = []
        i = 1
        for naca in FULL_NACAS:
            for aoa in AOAS:
                for re in RES:
                    cases.append({"id": f"case_{i:03d}", "naca": naca, "AoA": float(aoa), "Re": float(re)})
                    i += 1

    cfg = {
        "rho": 997.0,
        "nu": 1.0e-6,
        "p_inf": 101325.0,
        "p_vap": 2300.0,
        "chord": 1.0,
        "grid": {"nx": 128, "ny": 64, "x_min": -1.0, "x_max": 2.0, "y_min": -0.75, "y_max": 0.75},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
