#!/usr/bin/env python
"""Check that the paper-output figure/result archive is complete."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper_outputs" / "figures"
RES_DIR = ROOT / "paper_outputs" / "results"
MANIFEST = RES_DIR / "paper_figure_manifest.csv"

REQUIRED_RESULTS = [
    "generated_method_summary.csv",
    "generated_per_point_predictions.csv",
    "hnsw_k_sweep_reference.csv",
    "paper_reference_accuracy_latency_tradeoff.csv",
    "paper_reference_ann_architectures.csv",
    "paper_reference_transformer_architectures.csv",
    "paper_figure_manifest.csv",
]


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def main() -> None:
    if not MANIFEST.exists():
        fail("Missing paper figure manifest")
    manifest = pd.read_csv(MANIFEST)
    missing_rows = manifest[(manifest["exists"] != True) | (manifest["size_bytes"] <= 0)]
    missing_files = [p.name for p in FIG_DIR.iterdir()] if FIG_DIR.exists() else []
    if not missing_rows.empty:
        fail("Missing or empty figure files: " + ", ".join(missing_rows["figure_file"].tolist()))
    for name in manifest["figure_file"]:
        p = FIG_DIR / str(name)
        if not p.exists() or p.stat().st_size <= 0:
            fail(f"Manifest-listed figure is missing or empty: {name}")
    for name in REQUIRED_RESULTS:
        p = RES_DIR / name
        if not p.exists() or p.stat().st_size <= 0:
            fail(f"Missing or empty paper-output result: {name}")
    print(f"[OK] Paper-output archive complete: {len(manifest)} figure files and {len(REQUIRED_RESULTS)} result files checked.")


if __name__ == "__main__":
    main()
