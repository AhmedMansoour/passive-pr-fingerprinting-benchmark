# Reproducibility guide

This guide explains how to verify the repository, rerun executable benchmark components, and regenerate the manuscript-facing paper-output archive.

## 1. Environment

Recommended Python versions: 3.10--3.13 for the core benchmark. The core benchmark does not require GPU support.

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For the extended optional model stack:

```bash
pip install -r requirements-full.txt
```

TensorFlow/Keras paths are optional and can be installed separately with `requirements-tensorflow.txt` on supported Python/platform combinations.

## 2. Fast release verification

```bash
python scripts/verify_release.py
```

This command runs the repository health check, pytest suite, KWNN benchmark, bootstrap analysis, result summary, standard plot generation, and paper-output figure generation.

## 3. Repository health check only

```bash
python scripts/check_repository.py
```

This check verifies that:

- required data files exist;
- raw data can be parsed with the expected separator;
- the dataset contains 36 training/reference locations and 9 held-out test locations;
- six AP/monitor-node columns are present;
- original MAC addresses are not exposed;
- anonymized `Device_ID` values are present;
- Python source files compile;
- no Python bytecode/cache files are included.

## 4. Executable benchmark scripts

```bash
python experiments/run_kwnn_benchmark.py
python experiments/run_ensemble_benchmark.py
python experiments/run_mlp_benchmark.py
python experiments/run_transformer_benchmark.py
python experiments/run_latency_benchmark.py
python experiments/run_bootstrap_analysis.py
python experiments/run_summary.py
python plots/generate_all_plots.py
```

The default neural/attention scripts are configured for public-release verification so that they complete on ordinary CPU environments. Heavy all-architecture/all-seed neural runs can be requested explicitly:

```bash
python experiments/run_mlp_benchmark.py --full-neural
python experiments/run_transformer_benchmark.py --full-neural
```

The monitor-node density analysis is included with precomputed outputs because the full combinatorial sweep can take longer:

```bash
python experiments/run_ap_density_analysis.py
```

## 5. Paper-output archive

The manuscript-facing output archive is under:

```text
paper_outputs/
├── figures/
├── results/
└── logs/
```

To regenerate the figure archive and manuscript-named figure files:

```bash
python plots/generate_paper_figures.py
```

The manifest below confirms whether all manuscript figure filenames are present:

```text
paper_outputs/results/paper_figure_manifest.csv
```

The paper-output archive includes both generated per-point diagnostics and paper-reference aggregate values for result-heavy figures. This separation keeps the public release easy to verify while preserving the full manuscript-facing outputs.

## 6. Expected validation evidence

The validation logs supplied with this release are stored in:

```text
paper_outputs/logs/
```

The principal audit summary is:

```text
RELEASE_RUN_REPORT.md
```

## 7. Interpretation boundary

The benchmark is a controlled single-environment evaluation. It is suitable for reproducible within-testbed comparison but not for making unrestricted claims about general deployment robustness across buildings, devices, or layouts. Reported rankings and confidence intervals should be interpreted as benchmark-level evidence for the evaluated testbed.

## Paper-chart reproduction

The repository includes a chart-level reproduction layer that maps each submitted paper figure to its original notebook/script source and regenerates the manuscript-facing chart files. From the repository root, run:

```bash
python reproducibility/paper_charts/standalone_scripts/generate_all_paper_charts.py
python scripts/check_chart_reproduction.py
```

The regenerated figures are saved in `reproducibility/paper_charts/generated_figures/` and copied to `paper_outputs/figures/`. The provenance table is available in `reproducibility/paper_charts/CHART_SOURCE_MAP.csv`.

