# Release validation

This repository was checked in a clean working copy before packaging.

Validated commands:

```bash
python scripts/check_repository.py
python -m pytest -q -p no:cacheprovider
python experiments/run_kwnn_benchmark.py
python experiments/run_ensemble_benchmark.py
python experiments/run_mlp_benchmark.py
python experiments/run_transformer_benchmark.py
python experiments/run_latency_benchmark.py
python experiments/run_bootstrap_analysis.py
python experiments/run_summary.py
python plots/generate_all_plots.py
python plots/generate_paper_figures.py
```

Observed status:

- Repository health check: passed.
- Pytest suite: 5 passed.
- KWNN benchmark: completed successfully.
- Ensemble benchmark: completed successfully.
- MLP public-release verification: completed successfully; TensorFlow skipped when not installed.
- Transformer public-release verification: completed successfully; TensorFlow skipped when not installed.
- Latency benchmark: completed successfully; timing values are hardware dependent.
- Bootstrap and summary generation: completed successfully.
- Standard plots and paper-output figures: completed successfully.
- Paper figure manifest: 34/34 manuscript figure files present.

Notes:

- The executable verification path is intentionally CPU-friendly.
- Heavy all-seed/all-architecture neural runs are available through `--full-neural`.
- The manuscript-facing results and figures are archived under `paper_outputs/`.
- Optional package availability can vary by platform. TensorFlow is optional.
