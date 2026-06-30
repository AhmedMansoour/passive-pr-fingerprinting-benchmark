# Open-source release audit

## Changes applied

- Removed all committed `__pycache__` directories and `.pyc` files.
- Corrected documentation to match the actual dataset: 36 training/reference locations and 9 held-out test locations.
- Corrected documentation to match the actual file names: `data/raw/df_all.csv` and `data/raw/test_df.csv`.
- Anonymized the raw data by removing the original `MAC Address` column and replacing it with `Device_ID`.
- Added processed fingerprint matrices under `data/processed/` for transparent reuse.
- Added `DATASET_CARD.md` with scope, intended use, limitations, and privacy notes.
- Added `REPRODUCIBILITY.md` with check, smoke-test, and full-benchmark instructions.
- Added `scripts/check_repository.py` for release-quality repository validation.
- Added `tests/` with data-integrity and KWNN smoke tests.
- Added GitHub Actions CI under `.github/workflows/ci.yml`.
- Split dependencies into `requirements.txt`, `requirements-full.txt`, and `requirements-dev.txt`.
- Added `pyproject.toml`, `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md`.
- Updated `.gitignore` so reference results and selected plots can be tracked intentionally.
- Updated the repository license metadata and holder information.

## Critical release notes before uploading to GitHub

1. Replace `https://github.com/REPLACE_WITH_REPOSITORY_URL` in `CITATION.cff` with the real repository URL.
2. Confirm that CC BY 4.0 is the intended long-term license for both the code and dataset before public release.
3. After creating the GitHub repository, run the CI workflow and copy the repository link into the manuscript Data Availability statement.
4. Avoid making broad generalization claims in the repository text; the documentation now frames the benchmark as single-environment and reproducibility-oriented.
