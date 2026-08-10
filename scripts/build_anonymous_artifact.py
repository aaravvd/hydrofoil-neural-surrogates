#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".md", ".txt", ".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".csv"}
EXCLUDED_PARTS = {".git", ".venv", "output", "papers", "anonymous_artifact", "__pycache__", ".matplotlib-cache"}
REPLACEMENTS = {
    "Aarav Dixit": "Anonymous Authors",
    "aaravdixit": "anonymous",
    "aaravvd": "anonymous",
    "https://github.com/aaravvd/hydrofoil-neural-surrogates": "ANONYMOUS_REPOSITORY_URL",
    "10.5281/zenodo.21845241": "ANONYMOUS_DATA_URL",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a sanitized, sub-100 MB double-blind review artifact.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "anonymous_artifact")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "training_runs" / "revised")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "paper_results" / "revised")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "corrected_production" / "data" / "processed_grids")
    args = parser.parse_args()

    bundle = args.output_dir / "hydrofoil_surrogate_review_artifact"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    copy_source(bundle)
    copy_tree_sanitized(args.run_dir, bundle / "training_runs" / "revised")
    copy_tree_sanitized(args.result_dir, bundle / "paper_results" / "revised")
    copy_test_data(args.data_dir, bundle / "data" / "test_grids")
    write_readme(bundle)
    assert_anonymous(bundle)

    archive = args.output_dir / "hydrofoil_surrogate_review_artifact.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(args.output_dir))
    size_mb = archive.stat().st_size / 1024**2
    if size_mb >= 100:
        raise SystemExit(f"Artifact is {size_mb:.1f} MB; remove optional files before submission.")
    print(f"[ok] wrote anonymous artifact: {archive} ({size_mb:.1f} MB)")


def copy_source(bundle: Path) -> None:
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts[0] in {"corrected_production", "training_runs", "paper_results", "hso_results"}:
            continue
        if path.suffix not in TEXT_EXTENSIONS and path.name not in {"LICENSE", "Makefile"}:
            continue
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = path.read_text(errors="replace")
        for old, new in REPLACEMENTS.items():
            text = text.replace(old, new)
        destination.write_text(text)


def copy_tree_sanitized(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT_EXTENSIONS or path.suffix == ".csv":
            text = path.read_text(errors="replace")
            for old, new in REPLACEMENTS.items():
                text = text.replace(old, new)
            target.write_text(text)
        elif path.name == "best.pt":
            sanitize_checkpoint(path, target)
        else:
            shutil.copy2(path, target)


def sanitize_checkpoint(source: Path, target: Path) -> None:
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clean(item) for item in value)
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            for old, new in REPLACEMENTS.items():
                value = value.replace(old, new)
            if value.startswith(str(ROOT)):
                value = value.replace(str(ROOT), ".", 1)
        return value

    torch.save(clean(checkpoint), target)


def copy_test_data(data_dir: Path, destination: Path) -> None:
    manifest = json.loads((ROOT / "configs" / "family_split.json").read_text())
    test_families = set(manifest["test"])
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(data_dir.glob("case_*_grid.npz")):
        with np.load(path, allow_pickle=True) as data:
            if str(data["naca"]) not in test_families:
                continue
        shutil.copy2(path, destination / path.name)
        copied += 1
    if copied != 64:
        raise ValueError(f"Expected 64 untouched test grids, copied {copied}")


def write_readme(bundle: Path) -> None:
    (bundle / "ANONYMOUS_README.md").write_text(
        """# Anonymous review artifact

This artifact contains the executable training, final-test evaluation, optimization, and CFD validation code; the fixed family split; all three trained checkpoints per architecture; compact result files; and all 64 untouched test grids (NACA 0018 and NACA 4418). The test grids permit exact reproduction of every final predictive metric without revealing author identity.

The full 381-case CFD dataset is provided at `ANONYMOUS_DATA_URL` because it exceeds the conference supplementary-file limit. Until that anonymous URL is inserted, do not submit this archive as a complete data-release claim. Run `scripts/reproduce_paper.sh evaluate` to recompute final-test metrics from the included checkpoints and test grids after adjusting `DATA_DIR`, or follow `REPRODUCIBILITY.md` to regenerate CFD and retrain.

The family partition is frozen in `configs/family_split.json`. Test families are not used for fitting, early stopping, model selection, or hyperparameter selection.
"""
    )


def assert_anonymous(bundle: Path) -> None:
    forbidden = [key for key in REPLACEMENTS if key not in {"https://github.com/aaravvd/hydrofoil-neural-surrogates"}]
    hits = []
    for path in bundle.rglob("*"):
        if path.is_file() and path.suffix in TEXT_EXTENSIONS:
            text = path.read_text(errors="replace")
            hits.extend(f"{path.relative_to(bundle)}: {term}" for term in forbidden if term in text)
    if hits:
        raise ValueError("Identity leak(s) in anonymous artifact:\n" + "\n".join(hits))


if __name__ == "__main__":
    main()
