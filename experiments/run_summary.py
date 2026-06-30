#!/usr/bin/env python
"""Generate summary tables from all benchmark results."""
import os
import sys
import pandas as pd
import numpy as np
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    if not csv_files:
        print("No result files found. Run the benchmarks first.")
        return

    for csv_file in csv_files:
        name = os.path.basename(csv_file)
        if name == "all_methods_summary.csv":
            continue
        print(f"\n--- {name} ---")
        df = pd.read_csv(csv_file)
        print(df.to_string(index=False))

    # Merge accuracy results into a normalized summary schema when available.
    normalized = []

    def append_normalized(path, source):
        if not os.path.exists(path):
            return
        df = pd.read_csv(path)
        if "Method" in df.columns and "Mean_Error_m" in df.columns:
            out = pd.DataFrame({
                "method": df["Method"],
                "mean_m": df["Mean_Error_m"],
                "std_m": pd.Series(np.nan, index=df.index, dtype="float64"),
                "source": source,
            })
        elif {"method", "mean_m"}.issubset(df.columns):
            out = df[["method", "mean_m"]].copy()
            out["std_m"] = df["std_m"] if "std_m" in df.columns else pd.NA
            out["source"] = source
        else:
            return
        normalized.append(out)

    append_normalized(os.path.join(results_dir, "kwnn_benchmark.csv"), "kwnn")
    append_normalized(os.path.join(results_dir, "ensemble_benchmark.csv"), "ensemble")
    append_normalized(os.path.join(results_dir, "mlp_benchmark.csv"), "mlp")
    append_normalized(os.path.join(results_dir, "transformer_benchmark.csv"), "transformer")

    if normalized:
        combined = pd.concat(normalized, ignore_index=True)
        combined_path = os.path.join(results_dir, "all_methods_summary.csv")
        combined.to_csv(combined_path, index=False)
        print("\n--- Normalized Accuracy Summary ---")
        print(combined.to_string(index=False))
        print(f"\nSaved to {combined_path}")



if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
