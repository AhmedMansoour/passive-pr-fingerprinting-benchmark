#!/usr/bin/env python
"""Run MLP benchmark: sklearn, TensorFlow, and PyTorch improved architectures with multi-seed."""
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    MLP_ARCHITECTURES_SKLEARN, MLP_ARCHITECTURES_TF, MLP_TF_PARAMS,
    IMPROVED_MLP_CONFIGS, PYTORCH_TRAINING_PARAMS, QUICK_PYTORCH_TRAINING_PARAMS,
    RANDOM_SEEDS, QUICK_RANDOM_SEEDS, TRAIN_COORD_MAPPING, TEST_COORD_MAPPING,
)
from src.data_loader import load_dataset
from src.evaluation import compute_euclidean_errors, compute_error_statistics


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP benchmark.")
    parser.add_argument("--quick", action="store_true", help="Run one seed and one architecture per family for smoke verification.")
    parser.add_argument("--full-neural", action="store_true", help="Run all neural architectures and seeds; may be slow.")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = RANDOM_SEEDS if args.full_neural else QUICK_RANDOM_SEEDS
    if args.full_neural:
        sklearn_architectures = MLP_ARCHITECTURES_SKLEARN
        tf_architectures = MLP_ARCHITECTURES_TF
        pytorch_architectures = IMPROVED_MLP_CONFIGS
        pytorch_params = PYTORCH_TRAINING_PARAMS
    elif args.quick:
        sklearn_architectures = dict(list(MLP_ARCHITECTURES_SKLEARN.items())[:1])
        tf_architectures = dict(list(MLP_ARCHITECTURES_TF.items())[:1])
        pytorch_architectures = dict(list(IMPROVED_MLP_CONFIGS.items())[:1])
        pytorch_params = QUICK_PYTORCH_TRAINING_PARAMS
    else:
        # Default public-release run: compact and deterministic, while the
        # paper-reference aggregate neural results are archived separately.
        sklearn_architectures = dict(list(MLP_ARCHITECTURES_SKLEARN.items())[:2])
        tf_architectures = dict(list(MLP_ARCHITECTURES_TF.items())[:1])
        pytorch_architectures = dict(list(IMPROVED_MLP_CONFIGS.items())[:1])
        pytorch_params = QUICK_PYTORCH_TRAINING_PARAMS
    tf_params = dict(MLP_TF_PARAMS)
    if args.quick:
        tf_params["epochs"] = 5
    print("=" * 80)
    print("MLP BENCHMARK - sklearn, TensorFlow, PyTorch Improved")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    train_csv = os.path.join(data_dir, "df_all.csv")
    test_csv = os.path.join(data_dir, "test_df.csv")

    X_train, y_train, X_test, y_test, _ = load_dataset(
        train_csv, test_csv, TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, scale=True,
    )
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test\n")

    all_results = []

    # --- sklearn MLP (multi-seed) ---
    print("[sklearn MLP - multi-seed]")
    from src.models.mlp import train_mlp_sklearn
    for arch_name, layers in sklearn_architectures.items():
        seed_means = []
        for seed in seeds:
            model = train_mlp_sklearn(X_train, y_train, layers, random_state=seed)
            preds = model.predict(X_test)
            errors = compute_euclidean_errors(y_test, preds)
            stats = compute_error_statistics(errors)
            seed_means.append(stats["Mean_Error_m"])
        mean_m = np.mean(seed_means)
        std_m = np.std(seed_means)
        print(f"  {arch_name}: {mean_m:.3f} +/- {std_m:.3f} m")
        all_results.append({"method": f"sklearn_{arch_name}", "mean_m": mean_m, "std_m": std_m})

    # --- TensorFlow MLP (multi-seed) ---
    print("\n[TensorFlow MLP - multi-seed]")
    try:
        from src.models.mlp_tensorflow import train_mlp_tf
        import tensorflow as tf
        for arch_name, layers in tf_architectures.items():
            seed_means = []
            for seed in seeds:
                tf.random.set_seed(seed)
                np.random.seed(seed)
                model = train_mlp_tf(X_train, y_train, layers, seed=seed, **tf_params)
                preds = model.predict(X_test, verbose=0)
                errors = compute_euclidean_errors(y_test, preds)
                stats = compute_error_statistics(errors)
                seed_means.append(stats["Mean_Error_m"])
            mean_m = np.mean(seed_means)
            std_m = np.std(seed_means)
            print(f"  {arch_name}: {mean_m:.3f} +/- {std_m:.3f} m")
            all_results.append({"method": f"tf_{arch_name}", "mean_m": mean_m, "std_m": std_m})
    except ImportError:
        print("  [SKIP] TensorFlow not installed")

    # --- PyTorch Improved MLP (multi-seed) ---
    print("\n[PyTorch Improved MLP - multi-seed]")
    try:
        import torch
        from src.models.improved_models_pytorch import (
            ImprovedMLP, train_pytorch_model, prepare_pytorch_data,
        )
        params = pytorch_params
        for arch_name, layers in pytorch_architectures.items():
            seed_means = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                loader, X_te, y_te = prepare_pytorch_data(
                    X_train, y_train, X_test, y_test, batch_size=params["batch_size"],
                )
                model = ImprovedMLP(X_train.shape[1], layers, dropout_rate=params["dropout_rate"])
                model = train_pytorch_model(
                    model, loader, X_te, y_te,
                    epochs=params["epochs"], patience=params["patience"],
                    learning_rate=params["learning_rate"],
                    weight_decay=params["weight_decay"], grad_clip=params["grad_clip"],
                )
                model.eval()
                with torch.no_grad():
                    preds = model(X_te).cpu().numpy()
                errors = compute_euclidean_errors(y_test, preds)
                stats = compute_error_statistics(errors)
                seed_means.append(stats["Mean_Error_m"])
            mean_m = np.mean(seed_means)
            std_m = np.std(seed_means)
            print(f"  {arch_name}: {mean_m:.3f} +/- {std_m:.3f} m")
            all_results.append({"method": f"pytorch_{arch_name}", "mean_m": mean_m, "std_m": std_m})
    except ImportError:
        print("  [SKIP] PyTorch not installed")

    # Save
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_results)
    out_name = "mlp_benchmark.csv" if args.full_neural else ("mlp_benchmark_quick.csv" if args.quick else "mlp_benchmark_release.csv")
    df.to_csv(os.path.join(results_dir, out_name), index=False)
    print(f"\nResults saved to results/{out_name}")


if __name__ == "__main__":
    main()
