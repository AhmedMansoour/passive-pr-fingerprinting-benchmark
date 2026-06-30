#!/usr/bin/env python
"""Measure inference latency for all model families."""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    LATENCY_CONFIG, MLP_ARCHITECTURES_SKLEARN,
    RANDOM_FOREST_CONFIG, GRADIENT_BOOSTING_CONFIG,
    TRAIN_COORD_MAPPING, TEST_COORD_MAPPING,
)
from src.data_loader import load_dataset
from src.utils.latency import measure_latency
from src.models.kwnn import train_kwnn
from src.models.mlp import train_mlp_sklearn
from src.models.random_forest import train_random_forest
from src.models.gradient_boosting import train_gradient_boosting


def main():
    print("=" * 80)
    print("LATENCY BENCHMARK")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    X_train, y_train, X_test, y_test, _ = load_dataset(
        os.path.join(data_dir, "df_all.csv"),
        os.path.join(data_dir, "test_df.csv"),
        TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, scale=True,
    )

    test_input = X_test[0:1]  # Single sample
    num_reps = LATENCY_CONFIG["num_repetitions"]
    warmup = LATENCY_CONFIG["warmup_reps"]
    results = []

    # KWNN
    print("\n[KWNN]")
    for k in [3, 6]:
        for metric in ["euclidean", "manhattan"]:
            name = f"KWNN_{metric}_k{k}"
            model = train_kwnn(X_train, y_train, k, metric)
            lat = measure_latency(model, test_input, num_reps, warmup)
            lat["method"] = name
            results.append(lat)
            print(f"  {name}: {lat['mean_ms']:.4f} ms")

    # MLP (sklearn)
    print("\n[MLP sklearn]")
    for arch_name, layers in list(MLP_ARCHITECTURES_SKLEARN.items())[:2]:
        model = train_mlp_sklearn(X_train, y_train, layers, random_state=42)
        lat = measure_latency(model, test_input, num_reps, warmup)
        lat["method"] = arch_name
        results.append(lat)
        print(f"  {arch_name}: {lat['mean_ms']:.4f} ms")

    # Random Forest
    print("\n[Random Forest]")
    model = train_random_forest(X_train, y_train, random_state=42,
                                n_estimators=RANDOM_FOREST_CONFIG["n_estimators"],
                                n_jobs=RANDOM_FOREST_CONFIG["n_jobs"])
    lat = measure_latency(model, test_input, num_reps, warmup)
    lat["method"] = "RandomForest"
    results.append(lat)
    print(f"  RandomForest: {lat['mean_ms']:.4f} ms")

    # Gradient Boosting
    print("\n[Gradient Boosting]")
    model = train_gradient_boosting(X_train, y_train, random_state=42,
                                    n_estimators=GRADIENT_BOOSTING_CONFIG["n_estimators"])
    lat = measure_latency(model, test_input, num_reps, warmup)
    lat["method"] = "GradientBoosting"
    results.append(lat)
    print(f"  GradientBoosting: {lat['mean_ms']:.4f} ms")

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(results_dir, "latency_benchmark.csv"), index=False)
    print(f"\nSaved to results/latency_benchmark.csv")


if __name__ == "__main__":
    main()
