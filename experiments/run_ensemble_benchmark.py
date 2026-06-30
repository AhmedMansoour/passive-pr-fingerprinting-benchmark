#!/usr/bin/env python
"""Run ensemble method benchmark: RF, XGBoost, CatBoost, Gradient Boosting with multi-seed evaluation."""
import argparse
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    RANDOM_SEEDS, QUICK_RANDOM_SEEDS, RANDOM_FOREST_CONFIG, GRADIENT_BOOSTING_CONFIG,
    TRAIN_COORD_MAPPING, TEST_COORD_MAPPING,
)
from src.data_loader import load_dataset
from src.evaluation import compute_euclidean_errors, compute_error_statistics
from src.models.random_forest import train_random_forest
from src.models.gradient_boosting import (
    train_gradient_boosting, train_xgboost, train_xgboost_gridsearch, train_catboost,
)


def run_multiseed(name, train_fn, X_train, y_train, X_test, y_test, seeds, **kwargs):
    """Run a model across multiple seeds and report mean +/- std of mean error."""
    seed_means = []
    for seed in seeds:
        model = train_fn(X_train, y_train, random_state=seed, **kwargs)
        preds = model.predict(X_test)
        errors = compute_euclidean_errors(y_test, preds)
        stats = compute_error_statistics(errors)
        seed_means.append(stats["Mean_Error_m"])

    mean_of_means = np.mean(seed_means)
    std_of_means = np.std(seed_means)
    print(f"  {name}: {mean_of_means:.3f} +/- {std_of_means:.3f} m (over {len(seeds)} seeds)")
    return {"method": name, "mean_m": mean_of_means, "std_m": std_of_means}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run compact ensemble smoke verification.")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = QUICK_RANDOM_SEEDS if args.quick else RANDOM_SEEDS
    rf_estimators = 10 if args.quick else RANDOM_FOREST_CONFIG["n_estimators"]
    gb_estimators = 10 if args.quick else GRADIENT_BOOSTING_CONFIG["n_estimators"]
    print("=" * 80)
    print("ENSEMBLE BENCHMARK - RF, XGBoost, CatBoost, Gradient Boosting")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    train_csv = os.path.join(data_dir, "df_all.csv")
    test_csv = os.path.join(data_dir, "test_df.csv")

    X_train, y_train, X_test, y_test, _ = load_dataset(
        train_csv, test_csv, TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, scale=True,
    )
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test\n")

    results = []

    # Random Forest
    print("[Random Forest]")
    r = run_multiseed("RandomForest", train_random_forest, X_train, y_train, X_test, y_test,
                      seeds, n_estimators=rf_estimators,
                      n_jobs=1)
    results.append(r)

    # Gradient Boosting
    print("\n[Gradient Boosting]")
    r = run_multiseed("GradientBoosting", train_gradient_boosting, X_train, y_train, X_test, y_test,
                      seeds, n_estimators=gb_estimators)
    results.append(r)

    # XGBoost
    print("\n[XGBoost]")
    try:
        if args.quick:
            model = train_xgboost(
                X_train, y_train, random_state=42,
                n_estimators=10, max_depth=3, learning_rate=0.1,
            )
            preds = model.predict(X_test)
            errors = compute_euclidean_errors(y_test, preds)
            stats = compute_error_statistics(errors)
            print(f"  XGBoost quick: {stats['Mean_Error_m']:.3f} m")
            results.append({"method": "XGBoost_Quick", "mean_m": stats["Mean_Error_m"], "std_m": 0.0})
        else:
            best_model, best_params = train_xgboost_gridsearch(X_train, y_train)
            preds = best_model.predict(X_test)
            errors = compute_euclidean_errors(y_test, preds)
            stats = compute_error_statistics(errors)
            print(f"  XGBoost (best): {stats['Mean_Error_m']:.3f} m, params={best_params}")
            results.append({"method": "XGBoost_GridSearch", "mean_m": stats["Mean_Error_m"], "std_m": 0.0})
    except ImportError:
        print("  [SKIP] xgboost not installed")

    # CatBoost
    print("\n[CatBoost]")
    try:
        model = train_catboost(X_train, y_train, iterations=(10 if args.quick else 100))
        preds = model.predict(X_test)
        errors = compute_euclidean_errors(y_test, preds)
        stats = compute_error_statistics(errors)
        print(f"  CatBoost: {stats['Mean_Error_m']:.3f} m")
        results.append({"method": "CatBoost", "mean_m": stats["Mean_Error_m"], "std_m": 0.0})
    except ImportError:
        print("  [SKIP] catboost not installed")

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(results)
    out_name = "ensemble_benchmark_quick.csv" if args.quick else "ensemble_benchmark.csv"
    df.to_csv(os.path.join(results_dir, out_name), index=False)
    print(f"\nResults saved to results/{out_name}")


if __name__ == "__main__":
    main()
