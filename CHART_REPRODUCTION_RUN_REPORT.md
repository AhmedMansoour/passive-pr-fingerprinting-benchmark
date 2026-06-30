# Chart-reproduction validation report

This release was checked as a paper-chart reproduction package.

## Chart provenance

- Reference chart files checked: 23
- Chart-to-source provenance entries: 23
- Provenance table: `reproducibility/paper_charts/CHART_SOURCE_MAP.csv`

Validation command:

```bash
python scripts/check_chart_provenance.py
```

Observed result:

```text
Chart provenance check: PASSED (23 charts mapped)
```

## Chart regeneration

Main command:

```bash
python reproducibility/paper_charts/standalone_scripts/generate_all_paper_charts.py
```

The command regenerates the chart set and writes outputs to:

```text
reproducibility/paper_charts/generated_figures/
paper_outputs/figures/
```

Observed result:

```text
[OK] generated charts: 23 / 23
[OK] full chart reproduction completed
```

## Output validation

Validation command:

```bash
python scripts/check_chart_reproduction.py
python scripts/check_paper_outputs.py
```

Observed result:

```text
[OK] Chart reproduction archive complete: 23 generated paper charts checked.
[OK] Paper-output archive complete: 34 figure files and 7 result files checked.
```

## Practical reproducibility note

The standalone chart scripts use the released raw data and archived result assets from the original experiments. This keeps the paper figures reproducible without requiring reviewers to retrain all neural and attention models before verifying the chart outputs.
