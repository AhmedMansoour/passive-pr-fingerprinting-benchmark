#!/usr/bin/env python
"""Run benchmark entry points.

By default this script runs the lightweight core suite. Use --full to execute all
model-family scripts, including optional and potentially slower neural/ensemble
experiments.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

CORE_SCRIPTS = [
    "run_kwnn_benchmark.py",
    "run_bootstrap_analysis.py",
    "run_summary.py",
]

FULL_SCRIPTS = [
    "run_kwnn_benchmark.py",
    "run_ensemble_benchmark.py",
    "run_mlp_benchmark.py",
    "run_transformer_benchmark.py",
    "run_latency_benchmark.py",
    "run_bootstrap_analysis.py",
    "run_ap_density_analysis.py",
    "run_summary.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run passive PR benchmark scripts.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full optional benchmark suite. Default: lightweight core suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = FULL_SCRIPTS if args.full else CORE_SCRIPTS
    suite_name = "FULL OPTIONAL" if args.full else "CORE"

    print("=" * 80)
    print(f"RUNNING {suite_name} BENCHMARK SUITE")
    print("=" * 80)

    total_start = time.time()
    failed = []

    for script in scripts:
        path = os.path.join(EXPERIMENTS_DIR, script)
        if not os.path.exists(path):
            print(f"\n[SKIP] {script} — not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"[RUN] {script}")
        print("=" * 60)

        start = time.time()
        result = subprocess.run([sys.executable, path], capture_output=False)
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"[OK] {script} completed in {elapsed:.1f}s")
        else:
            print(f"[FAIL] {script} (exit code {result.returncode})")
            failed.append(script)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK SUITE COMPLETE — {total_elapsed:.1f}s total")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        raise SystemExit(1)
    print("All selected scripts completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
