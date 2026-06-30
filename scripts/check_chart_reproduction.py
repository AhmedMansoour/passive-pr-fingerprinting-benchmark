#!/usr/bin/env python
"""Validate generated paper-chart reproduction outputs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHART_ROOT = ROOT / "reproducibility" / "paper_charts"
STATUS = CHART_ROOT / "CHART_REPRODUCTION_STATUS.csv"
GENERATED = CHART_ROOT / "generated_figures"
FIG_DIR = ROOT / "paper_outputs" / "figures"
EXPECTED_COUNT = 23


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def main() -> None:
    if not STATUS.exists():
        fail("Missing CHART_REPRODUCTION_STATUS.csv. Run generate_all_paper_charts.py first.")
    df = pd.read_csv(STATUS)
    if len(df) != EXPECTED_COUNT:
        fail(f"Expected {EXPECTED_COUNT} chart-status rows, found {len(df)}")
    required_cols = {"chart_file", "reference_exists", "generated_exists", "paper_output_exists", "generated_size_bytes"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        fail("Status file is missing columns: " + ", ".join(sorted(missing_cols)))
    bad = df[(df["generated_exists"] != True) | (df["paper_output_exists"] != True) | (df["generated_size_bytes"] <= 0)]
    if not bad.empty:
        fail("Missing or empty generated charts: " + ", ".join(bad["chart_file"].astype(str).tolist()))
    for name in df["chart_file"].astype(str):
        for base in (GENERATED, FIG_DIR):
            p = base / name
            if not p.exists() or p.stat().st_size <= 0:
                fail(f"Missing or empty chart: {p.relative_to(ROOT)}")
    zero = [p.relative_to(ROOT) for p in list(GENERATED.glob("*")) + list(FIG_DIR.glob("*")) if p.is_file() and p.stat().st_size == 0]
    if zero:
        fail("Zero-byte chart files found: " + ", ".join(map(str, zero[:10])))
    print(f"[OK] Chart reproduction archive complete: {len(df)} generated paper charts checked.")


if __name__ == "__main__":
    main()
