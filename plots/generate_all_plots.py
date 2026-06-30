#!/usr/bin/env python
"""Generate all publication-quality figures from benchmark results."""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_kwnn_comparison():
    """Bar chart comparing KWNN metrics × k values."""
    path = os.path.join(RESULTS_DIR, "kwnn_benchmark.csv")
    if not os.path.exists(path):
        print("  [SKIP] kwnn_benchmark.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.set_facecolor("white")
    methods = df["Method"]
    errors = df["Mean_Error_m"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(range(len(methods)), errors, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Localization Error (m)")
    ax.set_title("KWNN: All Distance Metrics × k Values")
    for bar, val in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "kwnn_metric_comparison.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")


def plot_model_comparison():
    """4-panel comprehensive comparison."""
    summary_path = os.path.join(RESULTS_DIR, "all_methods_summary.csv")
    latency_path = os.path.join(RESULTS_DIR, "latency_benchmark.csv")
    if not os.path.exists(summary_path):
        print("  [SKIP] all_methods_summary.csv not found")
        return

    df = pd.read_csv(summary_path)
    df = df.dropna(subset=["method", "mean_m"]).copy()
    if df.empty:
        print("  [SKIP] all_methods_summary.csv has no normalized method/mean_m rows")
        return
    df["method"] = df["method"].astype(str)
    df["mean_m"] = pd.to_numeric(df["mean_m"], errors="coerce")
    df["std_m"] = pd.to_numeric(df.get("std_m", pd.Series([np.nan] * len(df))), errors="coerce")
    df = df.dropna(subset=["mean_m"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    for ax in axes.flat:
        ax.set_facecolor("white")

    # (a) Accuracy bar chart
    ax = axes[0, 0]
    methods = df["method"]
    mean_errors = df["mean_m"]
    std_errors = df["std_m"].fillna(0)
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    ax.barh(range(len(methods)), mean_errors, xerr=std_errors, color=colors, capsize=3)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel("Mean Error (m)")
    ax.set_title("(a) Localization Accuracy")
    ax.invert_yaxis()

    # (b) Latency (if available)
    ax = axes[0, 1]
    if os.path.exists(latency_path):
        lat_df = pd.read_csv(latency_path)
        ax.barh(range(len(lat_df)), lat_df["mean_ms"], color="steelblue")
        ax.set_yticks(range(len(lat_df)))
        ax.set_yticklabels(lat_df["method"], fontsize=7)
        ax.set_xlabel("Mean Latency (ms)")
        ax.set_xscale("log")
        ax.invert_yaxis()
    ax.set_title("(b) Inference Latency (log scale)")

    # (c) Accuracy vs Latency scatter
    ax = axes[1, 0]
    if os.path.exists(latency_path):
        lat_df = pd.read_csv(latency_path)
        # Match by method name (best effort)
        for _, row in lat_df.iterrows():
            token = str(row["method"]).split("_")[0]
            matching = df[df["method"].str.contains(token, case=False, na=False)]
            if len(matching) > 0:
                ax.scatter(row["mean_ms"], matching.iloc[0]["mean_m"], s=80, zorder=3)
                ax.annotate(row["method"], (row["mean_ms"], matching.iloc[0]["mean_m"]), fontsize=6)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Mean Error (m)")
    ax.set_title("(c) Accuracy vs. Latency Trade-off")

    # (d) Reproducibility ranking (by std)
    ax = axes[1, 1]
    if "std_m" in df.columns:
        sorted_df = df.dropna(subset=["std_m"]).sort_values("std_m")
        sorted_df = sorted_df[sorted_df["std_m"] > 0]
        if len(sorted_df) > 0:
            ax.barh(range(len(sorted_df)), sorted_df["std_m"], color="coral")
            ax.set_yticks(range(len(sorted_df)))
            ax.set_yticklabels(sorted_df["method"], fontsize=7)
            ax.set_xlabel("Std of Mean Error (m)")
            ax.invert_yaxis()
    ax.set_title("(d) Reproducibility (lower = better)")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "model_comparison_comprehensive.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")


def plot_ecdf():
    """ECDF of localization errors for key methods."""
    # This needs per-point errors which aren't in summary CSVs
    # Generate placeholder if kwnn_benchmark exists
    print("  [INFO] ECDF requires per-point errors — run individual benchmarks and collect per-point results.")


def plot_bootstrap_ci():
    """Bootstrap CI visualization."""
    path = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
    if not os.path.exists(path):
        print("  [SKIP] bootstrap_ci.csv not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    ax.set_facecolor("white")
    y_pos = range(len(df))
    ax.errorbar(
        df["Mean"], y_pos,
        xerr=[df["Mean"] - df["Mean_CI_Lower"], df["Mean_CI_Upper"] - df["Mean"]],
        fmt="o", color="steelblue", capsize=5, markersize=8,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Method"])
    ax.set_xlabel("Mean Localization Error (m)")
    ax.set_title("95% Bootstrap Confidence Intervals")
    ax.invert_yaxis()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "bootstrap_ci.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")


def main():
    print("=" * 80)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 80)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n[1] KWNN metric comparison")
    plot_kwnn_comparison()
    print("\n[2] Model comparison (4-panel)")
    plot_model_comparison()
    print("\n[3] ECDF")
    plot_ecdf()
    print("\n[4] Bootstrap CI")
    plot_bootstrap_ci()

    print("\n" + "=" * 80)
    print("DONE — figures saved to plots/")
    print("=" * 80)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
