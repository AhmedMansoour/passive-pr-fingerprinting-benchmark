#!/usr/bin/env python
"""Repository health checks for the passive PR fingerprinting benchmark."""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from configs.experiment_config import TRAIN_COORD_MAPPING, TEST_COORD_MAPPING

RAW_TRAIN = ROOT / "data" / "raw" / "df_all.csv"
RAW_TEST = ROOT / "data" / "raw" / "test_df.csv"
PROCESSED_TRAIN = ROOT / "data" / "processed" / "train_fingerprints.csv"
PROCESSED_TEST = ROOT / "data" / "processed" / "test_fingerprints.csv"
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def check_files() -> None:
    required = [
        RAW_TRAIN,
        RAW_TEST,
        PROCESSED_TRAIN,
        PROCESSED_TEST,
        ROOT / "README.md",
        ROOT / "REPRODUCIBILITY.md",
        ROOT / "DATASET_CARD.md",
        ROOT / "CITATION.cff",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        fail("Missing required files: " + ", ".join(missing))


def check_data() -> None:
    train = pd.read_csv(RAW_TRAIN, sep=";")
    test = pd.read_csv(RAW_TEST, sep=";")

    train_locations = set(train["Point"].str.lower().unique())
    test_locations = set(test["Folder"].str.lower().unique())
    train_map = set(TRAIN_COORD_MAPPING)
    test_map = set(TEST_COORD_MAPPING)

    if train_locations != train_map:
        fail(
            "Training coordinate map does not exactly match raw data. "
            f"Missing={sorted(train_locations - train_map)}, extra={sorted(train_map - train_locations)}"
        )
    if test_locations != test_map:
        fail(
            "Test coordinate map does not exactly match raw data. "
            f"Missing={sorted(test_locations - test_map)}, extra={sorted(test_map - test_locations)}"
        )
    if len(train_locations) != 36:
        fail(f"Expected 36 training locations, found {len(train_locations)}")
    if len(test_locations) != 9:
        fail(f"Expected 9 held-out test locations, found {len(test_locations)}")
    if train["AP"].nunique() != 6 or test["AP"].nunique() != 6:
        fail("Expected six AP/monitor-node identifiers in train and test raw data")
    if "MAC Address" in train.columns or "MAC Address" in test.columns:
        fail("Raw files must not expose the original MAC Address column")
    if "Device_ID" not in train.columns or "Device_ID" not in test.columns:
        fail("Raw files must contain anonymized Device_ID values")

    for p, expected_rows in [(PROCESSED_TRAIN, 36), (PROCESSED_TEST, 9)]:
        df = pd.read_csv(p)
        ap_cols = [c for c in df.columns if c.startswith("ap")]
        if len(df) != expected_rows:
            fail(f"{p.relative_to(ROOT)} should have {expected_rows} rows, found {len(df)}")
        if len(ap_cols) != 6:
            fail(f"{p.relative_to(ROOT)} should contain six AP columns, found {len(ap_cols)}")
        if not {"location_id", "X_mm", "Y_mm"}.issubset(df.columns):
            fail(f"{p.relative_to(ROOT)} is missing location_id/X_mm/Y_mm columns")
        if not np.isfinite(df[["X_mm", "Y_mm", *ap_cols]].to_numpy(dtype=float)).all():
            fail(f"{p.relative_to(ROOT)} contains non-finite numeric values")


def check_no_cache_artifacts() -> None:
    bad = [p for p in ROOT.rglob("*") if p.name in CACHE_NAMES or p.suffix == ".pyc"]
    if bad:
        sample = ", ".join(str(p.relative_to(ROOT)) for p in bad[:8])
        fail(f"Cache/bytecode files should not be committed: {sample}")


def check_compile() -> None:
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", "venv", "build", "dist"} for part in path.parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            fail(f"Python syntax error in {path.relative_to(ROOT)}: {exc}")


def main() -> None:
    check_files()
    check_data()
    check_no_cache_artifacts()
    check_compile()
    print("[OK] Repository checks passed.")


if __name__ == "__main__":
    main()
