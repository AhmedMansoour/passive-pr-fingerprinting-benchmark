#!/usr/bin/env python
"""Run the fast release-verification path used before publishing the repository."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}

COMMANDS = [
    [sys.executable, "-u", "scripts/check_repository.py"],
    [sys.executable, "-u", "scripts/check_chart_provenance.py"],
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_data_integrity.py", "tests/test_smoke_kwnn.py", "tests/test_optional_model_smoke.py::test_sklearn_random_forest_smoke"],
    [sys.executable, "-u", "scripts/check_chart_reproduction.py"],
    [sys.executable, "-u", "scripts/check_paper_outputs.py"],
]


def remove_caches() -> None:
    for path in sorted(ROOT.rglob("*"), reverse=True):
        if path.name in CACHE_NAMES and path.exists():
            shutil.rmtree(path, ignore_errors=True)
        elif path.suffix == ".pyc" and path.exists():
            path.unlink(missing_ok=True)


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("[RUN] " + " ".join(cmd))
    print("=" * 80)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        env[var] = "1"
    subprocess.run(cmd, cwd=ROOT, check=True, timeout=300, env=env)


def main() -> None:
    remove_caches()
    for cmd in COMMANDS:
        run(cmd)
    remove_caches()
    print("\n[OK] Fast release verification and paper-output archive checks completed successfully.")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
