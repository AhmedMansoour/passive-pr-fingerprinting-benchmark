# Identifier and legacy-file audit

The public release excludes internal legacy notebooks and raw chart-export folders that are not required for execution.

Removed/not included:

- `reproducibility/paper_charts/legacy_original_scripts/`
- `reproducibility/paper_charts/original_result_assets/df_all.csv`
- `reproducibility/paper_charts/original_result_assets/test_df.csv`

These files were development/provenance artifacts only. The executable benchmark and chart reproduction workflows use the anonymized public data under `data/`, paper-level result tables under `results/` and `paper_outputs/results/`, and standalone plotting scripts under `reproducibility/paper_charts/standalone_scripts/`.

Identifier check before packaging: no MAC-address pattern of the form `XX:XX:XX:XX:XX:XX` or `XX-XX-XX-XX-XX-XX` was found in text/code/CSV/notebook files after removal. Released raw data use pseudonymous `Device_ID` values and AP labels such as `ap1`--`ap6`.
