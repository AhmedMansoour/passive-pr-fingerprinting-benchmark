# Paper Outputs

This folder documents the manuscript-facing outputs included with the public repository.

## Figures

All figure filenames referenced by the manuscript are available in:

```text
paper_outputs/figures/
```

A machine-readable manifest is provided at:

```text
paper_outputs/results/paper_figure_manifest.csv
```

## Results

The main CSV outputs are:

- `generated_method_summary.csv` — generated method-level summary from executable public scripts.
- `generated_per_point_predictions.csv` — generated per-test-location predictions and errors for reproducible diagnostics.
- `hnsw_k_sweep_reference.csv` — k-sweep reference used for the scalable-retrieval trend figure.
- `paper_reference_accuracy_latency_tradeoff.csv` — manuscript accuracy-latency reference table.
- `paper_reference_ann_architectures.csv` — manuscript ANN architecture reference values.
- `paper_reference_transformer_architectures.csv` — manuscript Transformer architecture reference values.
- `paper_figure_manifest.csv` — figure-presence audit.

## Regeneration

To regenerate the paper-output archive, run:

```bash
python plots/generate_paper_figures.py
```

To run the complete fast validation path, run:

```bash
python scripts/verify_release.py
```

## Paper-chart reproduction

The repository includes a chart-level reproduction layer that maps each submitted paper figure to its original notebook/script source and regenerates the manuscript-facing chart files. From the repository root, run:

```bash
python reproducibility/paper_charts/standalone_scripts/generate_all_paper_charts.py
python scripts/check_chart_reproduction.py
```

The regenerated figures are saved in `reproducibility/paper_charts/generated_figures/` and copied to `paper_outputs/figures/`. The provenance table is available in `reproducibility/paper_charts/CHART_SOURCE_MAP.csv`.

