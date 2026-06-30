#!/usr/bin/env python
"""Run KWNN benchmark: all distance metrics × all k values."""
import os
import sys
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    KWNN_K_VALUES, KWNN_DISTANCE_METRICS, TRAIN_COORD_MAPPING, TEST_COORD_MAPPING,
)
from src.data_loader import load_dataset
from src.evaluation import compute_euclidean_errors, compute_error_statistics
from src.models.kwnn import run_kwnn_grid


def main():
    print("=" * 80)
    print("KWNN BENCHMARK — All Metrics × All k Values")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    train_csv = os.path.join(data_dir, "df_all.csv")
    test_csv = os.path.join(data_dir, "test_df.csv")

    X_train, y_train, X_test, y_test, _ = load_dataset(
        train_csv, test_csv, TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, scale=True,
    )
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X_train.shape[1]} APs\n")

    results = run_kwnn_grid(X_train, y_train, X_test, KWNN_K_VALUES, KWNN_DISTANCE_METRICS)

    # Compute errors and save results
    rows = []
    for r in results:
        errors = compute_euclidean_errors(y_test, r["y_pred"])
        stats = compute_error_statistics(errors)
        name = f"KWNN_{r['metric']}_k{r['k']}"
        print(f"  {name}: mean={stats['Mean_Error_m']:.3f}m, median={stats['Median_Error_m']:.3f}m")
        rows.append({
            "Method": name,
            "Mean_Error_m": stats["Mean_Error_m"],
            "Median_Error_m": stats["Median_Error_m"],
            "Max_Error_m": stats["Max_Error_m"],
        })

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "kwnn_benchmark.csv"), index=False)
    print(f"\nResults saved to results/kwnn_benchmark.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
