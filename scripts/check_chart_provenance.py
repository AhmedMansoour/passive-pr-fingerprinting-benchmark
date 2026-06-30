#!/usr/bin/env python
"""Validate that each manuscript chart has a reference figure and provenance entry."""
from pathlib import Path
import csv, sys
ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "reproducibility" / "paper_charts"
manifest = base / "CHART_SOURCE_MAP.csv"
refs = base / "reference_figures"
if not manifest.exists():
    raise SystemExit("Missing CHART_SOURCE_MAP.csv")
rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
missing=[]
for row in rows:
    p=refs / row["chart_file"]
    if not p.exists() or p.stat().st_size == 0:
        missing.append(row["chart_file"])
    if not (row["provenance_source"] or row["adapted_script_added"]):
        missing.append(row["chart_file"]+" (no source)")
if missing:
    raise SystemExit("Missing/invalid chart provenance: " + ", ".join(missing))
print(f"Chart provenance check: PASSED ({len(rows)} charts mapped)")
