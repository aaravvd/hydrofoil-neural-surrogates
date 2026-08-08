from pathlib import Path
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", type=Path, default=Path("data/processed_grids/case_001_grid.npz"))
args = parser.parse_args()

data = np.load(args.path)

fields = [
    "Ux","Uy","p","nut","k","omega",
    "Rxx","Rxy","Ryy","Cp","cavitation_margin",
    "cavitation_indicator","fluid_mask"
]

print("file:", args.path)
print("source:", data["source"] if "source" in data.files else "unknown")
print("naca:", data["naca"] if "naca" in data.files else "unknown")
print()

for field in fields:
    if field not in data.files:
        print(field)
        print("missing")
        print()
        continue

    arr = data[field]

    print(field)
    print("shape:", arr.shape)
    print("NaNs:", np.isnan(arr).sum())
    print("Finite:", np.isfinite(arr).sum())
    print("Min:", np.nanmin(arr))
    print("Max:", np.nanmax(arr))
    if field == "cavitation_indicator":
        print("Positive cells:", int(np.nansum(arr)))
    print()
