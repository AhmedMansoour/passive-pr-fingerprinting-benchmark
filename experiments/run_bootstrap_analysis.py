#!/usr/bin/env python
"""Compute bootstrap confidence intervals for main methods."""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    BOOTSTRAP_CONFIG, KWNN_K_VALUES,
    RANDOM_FOREST_CONFIG, RANDOM_SEEDS,
    TRAIN_COORD_MAPPING, TEST_COORD_MAPPING,
)
from src.data_loader import load_dataset
from src.evaluation import compute_euclidean_errors
from src.utils.bootstrap import bootstrap_ci, print_bootstrap_summary
from src.models.kwnn import train_kwnn
from src.models.random_forest import train_random_forest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a compact configuration for release verification.")
    args = parser.parse_args()

    print("=" * 80)
    print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    X_train, y_train, X_test, y_test, _ = load_dataset(
        os.path.join(data_dir, "df_all.csv"),
        os.path.join(data_dir, "test_df.csv"),
        TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, scale=True,
    )

    ci_results = []

    # Best KWNN
    model = train_kwnn(X_train, y_train, k=5, metric="euclidean")
    preds = model.predict(X_test)
    errors = compute_euclidean_errors(y_test, preds) / 1000  # meters
    bootstrap_config = dict(BOOTSTRAP_CONFIG)
    rf_config = dict(RANDOM_FOREST_CONFIG)
    seeds = list(RANDOM_SEEDS)
    if args.quick:
        bootstrap_config["num_samples"] = min(1000, int(bootstrap_config.get("num_samples", 1000)))
        rf_config["n_estimators"] = min(20, int(rf_config.get("n_estimators", 20)))
        rf_config["n_jobs"] = 1
        seeds = [RANDOM_SEEDS[0]]

    ci = bootstrap_ci(errors, **bootstrap_config)
    print_bootstrap_summary("KWNN_euclidean_k5", ci)
    ci["method"] = "KWNN_euclidean_k5"
    ci_results.append(ci)

    # Random Forest (multi-seed pooled errors)
    all_errors = []
    for seed in seeds:
        model = train_random_forest(X_train, y_train, random_state=seed, **rf_config)
        preds = model.predict(X_test)
        errors = compute_euclidean_errors(y_test, preds) / 1000
        all_errors.extend(errors.tolist())
    ci = bootstrap_ci(np.array(all_errors), **bootstrap_config)
    print_bootstrap_summary("RandomForest_pooled", ci)
    ci["method"] = "RandomForest_pooled"
    ci_results.append(ci)

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for c in ci_results:
        rows.append({
            "Method": c["method"],
            "Mean": c["mean"], "Mean_CI_Lower": c["mean_ci_lower"], "Mean_CI_Upper": c["mean_ci_upper"],
            "Median": c["median"], "Median_CI_Lower": c["median_ci_lower"], "Median_CI_Upper": c["median_ci_upper"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, "bootstrap_ci.csv"), index=False)
    print(f"\nSaved to results/bootstrap_ci.csv")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
