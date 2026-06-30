# Paper-chart reproduction package

This folder documents and reproduces the figure set used in the JNCA passive Wi-Fi probe-request fingerprinting manuscript.
It was built by auditing the submitted `Paper-Charts.zip` against the author's original plotting materials and then converting the relevant plotting logic into a reviewer-facing, standalone reproduction layer. The original legacy notebooks and raw chart-export CSVs are not included in the public release because they are not required for execution and may contain acquisition-stage identifiers.

## Folder contents

- `reference_figures/` - exact reference chart files from the paper chart package, plus the two additional submitted composite figures.
- `source_cells/` - extracted notebook/script cells used to identify the provenance and style of each chart.
- `original_result_assets/` - anonymized paper-level result assets used by chart scripts; raw legacy CSV exports are not included.
- `adapted_scripts/` - focused scripts for figures that were not directly recoverable as standalone legacy scripts.
- `standalone_scripts/generate_all_paper_charts.py` - one-command chart reproduction script.
- `generated_figures/` - audit copy of regenerated charts.
- `CHART_SOURCE_MAP.csv` and `CHART_SOURCE_MAP.md` - chart-by-chart provenance map.
- `CHART_REPRODUCTION_STATUS.csv` - generated/reference status table written by the reproduction script.

## One-command reproduction

From the repository root, run:

```bash
python reproducibility/paper_charts/standalone_scripts/generate_all_paper_charts.py
```

The script regenerates 23 paper-chart files, writes an audit copy to:

```text
reproducibility/paper_charts/generated_figures/
```

and also copies the manuscript-facing outputs to:

```text
paper_outputs/figures/
```

To validate the result, run:

```bash
python scripts/check_chart_reproduction.py
```

## Provenance validation

To verify that every paper chart has a mapped reference file and source/provenance entry, run:

```bash
python scripts/check_chart_provenance.py
```

## Reproducibility note

The chart-generation layer uses the released anonymized data and archived per-method/per-architecture result assets. This preserves the visual style and numerical content of the submitted figures while avoiding hardware-dependent retraining of all neural and attention models during routine repository verification.
