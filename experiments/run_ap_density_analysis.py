#!/usr/bin/env python
"""
Monitor-Node Density and Deployment-Geometry Sensitivity Analysis.

Rebuilds the radio map using subsets of the 6 monitor nodes (APs),
runs 3 representative models on each configuration, and reports
how localization performance changes with receiver density and geometry.

Addresses Reviewer 10: AP density trade-offs and network-level insight.
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.experiment_config import (
    TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, DEFAULT_RSSI, CSV_SEPARATOR,
    PYTORCH_TRAINING_PARAMS,
)
from src.data_loader import load_raw_data, build_fingerprint_matrix
from src.evaluation import compute_euclidean_errors, compute_error_statistics
from src.models.kwnn import train_kwnn
from src.models.random_forest import train_random_forest

# =========================================================================
# AP PHYSICAL POSITIONS (meters, read from xyaps.png deployment layout)
# =========================================================================
AP_POSITIONS = {
    "ap1": (24.0, 8.0),  # far right, top
    "ap2": (22.0, 5.0),  # right, mid
    "ap3": (12.0, 5.0),  # center, mid
    "ap4": (12.0, 7.0),  # center, upper
    "ap5": (3.0, 1.0),   # far left, bottom
    "ap6": (7.0, 5.0),   # left-center, mid
}

ALL_APS = sorted(AP_POSITIONS.keys())  # ap1..ap6
TRANSFORMER_SEEDS = [42, 123, 456]


# =========================================================================
# DISPERSION SCORE
# =========================================================================
def compute_dispersion(ap_subset):
    """Mean pairwise Euclidean distance among AP positions in the subset."""
    positions = [AP_POSITIONS[ap] for ap in ap_subset]
    if len(positions) < 2:
        return 0.0
    dists = []
    for (x1, y1), (x2, y2) in itertools.combinations(positions, 2):
        dists.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    return np.mean(dists)


# =========================================================================
# SUBSET GENERATION
# =========================================================================
def generate_all_subsets():
    """Generate all C(6,k) subsets for k=2..6 with dispersion scores."""
    ap_indices = list(range(1, 7))  # 1..6
    subsets = []
    for k in range(2, 7):
        for combo in itertools.combinations(ap_indices, k):
            ap_names = tuple(f"ap{i}" for i in combo)
            disp = compute_dispersion(ap_names)
            subsets.append({
                "k": k,
                "ap_indices": combo,
                "ap_names": ap_names,
                "dispersion": disp,
            })
    return subsets


def select_core_configurations(all_subsets):
    """Select the 12 core configurations for the main paper."""
    core = []
    for k in range(2, 7):
        k_subsets = sorted(
            [s for s in all_subsets if s["k"] == k],
            key=lambda x: x["dispersion"],
        )
        if k == 6:
            # Only one: full deployment
            core.append({**k_subsets[0], "geometry": "full"})
        elif k == 2:
            # Nearest pair and farthest pair
            core.append({**k_subsets[0], "geometry": "clustered"})
            core.append({**k_subsets[-1], "geometry": "dispersed"})
        else:
            # Clustered, median, dispersed
            core.append({**k_subsets[0], "geometry": "clustered"})
            mid = len(k_subsets) // 2
            core.append({**k_subsets[mid], "geometry": "median"})
            core.append({**k_subsets[-1], "geometry": "dispersed"})
    return core


# =========================================================================
# DATA LOADING WITH AP SUBSET
# =========================================================================
def extract_subset_data(train_pivot, test_pivot, ap_names, standardize=True):
    """Extract feature matrices for a given AP subset."""
    ap_cols = sorted(list(ap_names))
    X_train = train_pivot[ap_cols].values.astype(np.float64)
    y_train = train_pivot[["X", "Y"]].values.astype(np.float64)
    X_test = test_pivot[ap_cols].values.astype(np.float64)
    y_test = test_pivot[["X", "Y"]].values.astype(np.float64)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler


# =========================================================================
# MODEL RUNNERS
# =========================================================================
def run_kwnn(X_train, y_train, X_test):
    """KWNN k=3, euclidean (best retrieval config)."""
    model = train_kwnn(X_train, y_train, k=3, metric="euclidean")
    return model.predict(X_test)


def run_rf(X_train, y_train, X_test):
    """Random Forest n=100, seed=42."""
    model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42, n_jobs=-1)
    return model.predict(X_test)


def run_transformer(X_train, y_train, X_test, y_test):
    """Joint AE+Transformer (best config from paper), averaged over 3 seeds."""
    try:
        import torch
        from src.models.improved_models_pytorch import train_ae_transformer
    except ImportError:
        return None

    all_preds = []
    for seed in TRANSFORMER_SEEDS:
        preds = train_ae_transformer(
            X_train, y_train, X_test, y_test,
            bottleneck_dim=16, d_model=128, nhead=4, num_layers=2,
            dropout=0.0, lambda_recon=0.5, noise_std=0.1,
            epochs=60, batch_size=16, lr=1e-3, weight_decay=0.0,
            scheduler_T0=10, scheduler_Tmult=3, scheduler_eta_min=1e-5,
            val_frac=0.15, es_patience=8, seed=seed,
        )
        all_preds.append(preds)

    return np.mean(all_preds, axis=0)


# =========================================================================
# FIGURE GENERATION
# =========================================================================
def generate_figures(full_df, core_df, plots_dir):
    """Generate 2-panel figure for the main paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "font.size": 10, "axes.grid": True,
        "grid.alpha": 0.3,
    })

    models = ["KWNN", "RF", "Transformer"]
    colors = {"KWNN": "#1f77b4", "RF": "#2ca02c", "Transformer": "#d62728"}
    markers = {"KWNN": "o", "RF": "s", "Transformer": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    # --- Panel (a): Density sensitivity line plot ---
    ax = axes[0]
    for model_name in models:
        mdf = full_df[full_df["model"] == model_name]
        if mdf.empty:
            continue
        k_vals = sorted(mdf["k"].unique())
        means, mins, maxs = [], [], []
        for k in k_vals:
            kdf = mdf[mdf["k"] == k]
            means.append(kdf["Mean_Error_m"].mean())
            mins.append(kdf["Mean_Error_m"].min())
            maxs.append(kdf["Mean_Error_m"].max())
        means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
        ax.plot(k_vals, means, marker=markers[model_name], color=colors[model_name],
                label=model_name, linewidth=2, markersize=7)
        ax.fill_between(k_vals, mins, maxs, alpha=0.15, color=colors[model_name])

    ax.set_xlabel("Number of Monitor Nodes ($k$)", fontsize=11)
    ax.set_ylabel("Mean Localization Error (m)", fontsize=11)
    ax.set_title("(a) Effect of Monitor-Node Density", fontsize=12)
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.legend(fontsize=9)

    # --- Panel (b): Geometry bar chart at k=4 ---
    ax = axes[1]
    k4 = core_df[(core_df["k"] == 4)]
    if not k4.empty:
        geo_order = ["clustered", "median", "dispersed"]
        x = np.arange(len(geo_order))
        width = 0.22
        for i, model_name in enumerate(models):
            vals = []
            for geo in geo_order:
                row = k4[(k4["model"] == model_name) & (k4["geometry"] == geo)]
                vals.append(row["Mean_Error_m"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=model_name,
                   color=colors[model_name], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x + width)
        ax.set_xticklabels([g.capitalize() for g in geo_order], fontsize=10)
        ax.set_ylabel("Mean Localization Error (m)", fontsize=11)
        ax.set_title("(b) Effect of Deployment Geometry ($k=4$)", fontsize=12)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(plots_dir, "ap_density_sensitivity.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"  [OK] Figure saved: {out}")
    return out


# =========================================================================
# LATEX TABLE GENERATION
# =========================================================================
def print_latex_table(core_df):
    """Print LaTeX-formatted core results table."""
    print("\n% === LaTeX Table (copy to paper) ===")
    print(r"\begin{table}[!ht]")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \caption{Monitor-node density and deployment-geometry sensitivity.")
    print(r"    Mean localization error (m) for three representative model families")
    print(r"    across 12 subset configurations. $G(S)$: mean pairwise inter-node distance.}")
    print(r"  \label{tab:density_sensitivity}")
    print(r"  \scriptsize")
    print(r"  \renewcommand{\arraystretch}{1.1}")
    print(r"  \begin{tabular}{clccccc}")
    print(r"    \toprule")
    print(r"    $k$ & Geometry & $G(S)$ (m) & APs & KWNN & RF & Transformer \\")
    print(r"    \midrule")

    # Group by subset configuration
    seen = set()
    for _, row in core_df.iterrows():
        key = (row["k"], row["geometry"])
        if key in seen:
            continue
        seen.add(key)

        k = int(row["k"])
        geo = row["geometry"]
        disp = row["dispersion"]
        aps = row["ap_names"]

        # Get all three models for this configuration
        cfg_rows = core_df[(core_df["k"] == k) & (core_df["geometry"] == geo)]
        kwnn_val = cfg_rows[cfg_rows["model"] == "KWNN"]["Mean_Error_m"].values
        rf_val = cfg_rows[cfg_rows["model"] == "RF"]["Mean_Error_m"].values
        tf_val = cfg_rows[cfg_rows["model"] == "Transformer"]["Mean_Error_m"].values

        kwnn_str = f"{kwnn_val[0]:.3f}" if len(kwnn_val) > 0 else "--"
        rf_str = f"{rf_val[0]:.3f}" if len(rf_val) > 0 else "--"
        tf_str = f"{tf_val[0]:.3f}" if len(tf_val) > 0 else "--"

        ap_str = ",".join(str(a) for a in aps) if isinstance(aps, (list, tuple)) else str(aps)
        print(f"    {k} & {geo.capitalize()} & {disp:.1f} & {{{ap_str}}} & {kwnn_str} & {rf_str} & {tf_str} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 80)
    print("MONITOR-NODE DENSITY AND DEPLOYMENT-GEOMETRY SENSITIVITY ANALYSIS")
    print("=" * 80)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "raw")
    results_dir = os.path.join(project_root, "results")
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load data ONCE ---
    print("\n[1/5] Loading data...")
    train_raw, test_raw = load_raw_data(data_dir)
    train_pivot, test_pivot = build_fingerprint_matrix(train_raw, test_raw)
    print(f"  Train: {len(train_pivot)} points, Test: {len(test_pivot)} points")
    print(f"  APs available: {sorted([c for c in train_pivot.columns if c.startswith('ap')])}")

    # --- Generate subsets ---
    print("\n[2/5] Generating AP subsets...")
    all_subsets = generate_all_subsets()
    core_configs = select_core_configurations(all_subsets)
    print(f"  Total subsets: {len(all_subsets)}")
    print(f"  Core configurations: {len(core_configs)}")

    # Print core configurations
    for cfg in core_configs:
        print(f"    k={cfg['k']}, {cfg['geometry']:10s}, G={cfg['dispersion']:.1f}m, "
              f"APs={cfg['ap_names']}")

    # --- Run experiments ---
    print(f"\n[3/5] Running experiments ({len(all_subsets)} subsets x 3 models)...")
    all_results = []

    for i, subset in enumerate(all_subsets):
        k = subset["k"]
        ap_names = subset["ap_names"]
        disp = subset["dispersion"]
        is_core = any(
            c["k"] == k and c["ap_names"] == ap_names for c in core_configs
        )
        geo_label = ""
        for c in core_configs:
            if c["k"] == k and c["ap_names"] == ap_names:
                geo_label = c["geometry"]
                break

        # Extract data for this AP subset
        X_train, y_train, X_test, y_test, _ = extract_subset_data(
            train_pivot, test_pivot, ap_names, standardize=True
        )

        progress = f"[{i+1}/{len(all_subsets)}] k={k}, APs={ap_names}, G={disp:.1f}m"

        # --- KWNN ---
        preds = run_kwnn(X_train, y_train, X_test)
        errors = compute_euclidean_errors(y_test, preds)
        stats = compute_error_statistics(errors)
        all_results.append({
            "k": k, "ap_indices": subset["ap_indices"], "ap_names": ap_names,
            "dispersion": disp, "geometry": geo_label, "is_core": is_core,
            "model": "KWNN", **stats,
        })

        # --- Random Forest ---
        preds = run_rf(X_train, y_train, X_test)
        errors = compute_euclidean_errors(y_test, preds)
        stats = compute_error_statistics(errors)
        all_results.append({
            "k": k, "ap_indices": subset["ap_indices"], "ap_names": ap_names,
            "dispersion": disp, "geometry": geo_label, "is_core": is_core,
            "model": "RF", **stats,
        })

        # --- Transformer ---
        preds = run_transformer(X_train, y_train, X_test, y_test)
        if preds is not None:
            errors = compute_euclidean_errors(y_test, preds)
            stats = compute_error_statistics(errors)
            all_results.append({
                "k": k, "ap_indices": subset["ap_indices"], "ap_names": ap_names,
                "dispersion": disp, "geometry": geo_label, "is_core": is_core,
                "model": "Transformer", **stats,
            })
        else:
            print(f"  {progress} — Transformer SKIPPED (PyTorch not available)")

        if (i + 1) % 10 == 0 or is_core:
            last_kwnn = [r for r in all_results if r["k"] == k and
                         r["ap_names"] == ap_names and r["model"] == "KWNN"][-1]
            print(f"  {progress} — KWNN={last_kwnn['Mean_Error_m']:.3f}m")

    # --- Save results ---
    print(f"\n[4/5] Saving results...")
    full_df = pd.DataFrame(all_results)

    # Convert tuple columns to strings for CSV compatibility
    full_df["ap_indices_str"] = full_df["ap_indices"].apply(str)
    full_df["ap_names_str"] = full_df["ap_names"].apply(str)

    full_csv = os.path.join(results_dir, "ap_density_full.csv")
    full_df.drop(columns=["ap_indices", "ap_names"]).to_csv(full_csv, index=False)
    print(f"  [OK] Full results: {full_csv} ({len(full_df)} rows)")

    core_df = full_df[full_df["is_core"]].copy()
    core_csv = os.path.join(results_dir, "ap_density_core.csv")
    core_df.drop(columns=["ap_indices", "ap_names"]).to_csv(core_csv, index=False)
    print(f"  [OK] Core results: {core_csv} ({len(core_df)} rows)")

    # --- Summary ---
    print(f"\n[5/5] Summary by node count...")
    print(f"\n{'k':>3} | {'Model':>12} | {'Avg Mean (m)':>12} | {'Best (m)':>8} | {'Worst (m)':>9}")
    print("-" * 60)
    for k in range(2, 7):
        for model in ["KWNN", "RF", "Transformer"]:
            mdf = full_df[(full_df["k"] == k) & (full_df["model"] == model)]
            if mdf.empty:
                continue
            avg = mdf["Mean_Error_m"].mean()
            best = mdf["Mean_Error_m"].min()
            worst = mdf["Mean_Error_m"].max()
            print(f"{k:>3} | {model:>12} | {avg:>12.3f} | {best:>8.3f} | {worst:>9.3f}")
        if k < 6:
            print("-" * 60)

    # --- Generate figures ---
    print("\nGenerating figures...")
    generate_figures(full_df, core_df, plots_dir)

    # --- LaTeX table ---
    print_latex_table(core_df)

    print("\n" + "=" * 80)
    print("[DONE] AP DENSITY SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
