# Release Run Report

This repository was prepared as a public reproduction package for the JNCA manuscript:

**An End-to-End Pipeline for Passive Wi-Fi Probe Request Fingerprinting-Based Localization with Comparative Evaluation from Classical Retrieval to Attention-Based Models**

## Validation status

The repository was checked from a clean working tree after removing cache artifacts.

| Check | Status | Evidence |
|---|---:|---|
| Repository health check | PASSED | `paper_outputs/logs/check_repository_final.log` |
| Unit/smoke tests | PASSED | `paper_outputs/logs/pytest.log` |
| KWNN benchmark | PASSED | `paper_outputs/logs/run_kwnn_benchmark.log` |
| Ensemble benchmark | PASSED | `paper_outputs/logs/run_ensemble_benchmark.log` |
| MLP release verification | PASSED | `paper_outputs/logs/run_mlp_benchmark.log` |
| Transformer release verification | PASSED | `paper_outputs/logs/run_transformer_benchmark.log` |
| Latency benchmark | PASSED | `paper_outputs/logs/run_latency_benchmark.log` |
| Bootstrap analysis | PASSED | `paper_outputs/logs/run_bootstrap_analysis.log` |
| Summary table generation | PASSED | `paper_outputs/logs/run_summary.log` |
| Standard plot generation | PASSED | `paper_outputs/logs/generate_all_plots.log` |
| Paper-output figure archive | PASSED | `paper_outputs/logs/generate_paper_figures_stdout.log` |

## Main commands

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

A single convenience command is also provided:

```bash
python scripts/verify_release.py
```

## Paper-output archive

The directory `paper_outputs/` contains the reviewer-facing archive of results and figures.

```text
paper_outputs/
├── figures/   # 34 manuscript figure files, using manuscript filenames
├── results/   # generated and paper-reference CSV files
└── logs/      # validation and generation logs
```

The file `paper_outputs/results/paper_figure_manifest.csv` confirms that all manuscript figure filenames are present.

## Important reproducibility note

The repository contains two complementary reproduction layers:

1. **Executable release verification**: compact scripts that run reliably on ordinary CPU environments and verify data loading, benchmark execution, result generation, plotting, and tests.
2. **Paper-output archive**: the result tables and figure files corresponding to the submitted manuscript, including reference aggregate values for heavy neural/attention experiments.

This separation is intentional. Some neural and attention experiments are hardware- and package-version-sensitive. The compact release scripts provide a robust reviewer check, while the paper-output archive preserves the full manuscript outputs in stable form.

TensorFlow is optional. In the validation environment used here, TensorFlow was not installed, so TensorFlow branches were skipped as expected. PyTorch branches and all core CPU benchmarks were executed.

## Data privacy and release hygiene

- The original MAC-address field was removed.
- The released data use anonymized `Device_ID` values.
- No Python bytecode or cache folders are included in the final package.
- The dataset includes 36 training/reference locations and 9 spatially separated held-out test locations.
