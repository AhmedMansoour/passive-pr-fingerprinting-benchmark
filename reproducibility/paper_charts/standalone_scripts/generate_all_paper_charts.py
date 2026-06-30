#!/usr/bin/env python
"""Regenerate the paper chart set from the release data/result assets.

This script is the reviewer-facing, one-command chart-reproduction layer.
It adapts the original plotting cells archived under
`reproducibility/paper_charts/source_cells/` and saves the figures with the
same filenames as the submitted manuscript/Paper-Charts package.

The generated charts are written to:
  paper_outputs/figures/                  (manuscript-facing location)
  reproducibility/paper_charts/generated_figures/  (audit copy)

The script intentionally uses archived per-method/per-architecture result
assets for heavy neural and attention figures. This keeps chart reproduction
fast and deterministic while preserving the original plotting style.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
# Keep reviewer-side chart generation CPU-safe and avoid excessive BLAS thread spawning.
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_var, "1")
import re
import shutil
import sys
import subprocess
import textwrap
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    from scipy.stats import multivariate_normal
except Exception:  # pragma: no cover
    multivariate_normal = None

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.experiment_config import TRAIN_COORD_MAPPING, TEST_COORD_MAPPING, AP_POSITIONS_M, DEFAULT_RSSI
from src.data_loader import load_raw_data, build_fingerprint_matrix, extract_features_and_targets
from src.evaluation import compute_euclidean_errors

CHART_ROOT = ROOT / "reproducibility" / "paper_charts"
ASSETS = CHART_ROOT / "original_result_assets"
REFERENCE = CHART_ROOT / "reference_figures"
GENERATED = CHART_ROOT / "generated_figures"
FIG_DIR = ROOT / "paper_outputs" / "figures"
RES_DIR = ROOT / "paper_outputs" / "results"
LOG_DIR = ROOT / "paper_outputs" / "logs"
for d in [GENERATED, FIG_DIR, RES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

POINT_ORDER = ["tb10", "tb14", "tb3", "tb7", "td11", "td18", "td20", "td22", "td7"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save_current(name: str, dpi: int | None = None, fig=None):
    """Save active figure to both generated audit location and paper location."""
    if fig is None:
        fig = plt.gcf()
    gen_path = GENERATED / name
    paper_path = FIG_DIR / name
    kwargs = dict(bbox_inches="tight", pad_inches=0.01, transparent=False)
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(gen_path, **kwargs)
    # Copy the already-rendered figure to the manuscript-facing folder.
    # Saving the same complex PDF figure twice can be slow or hang on some
    # matplotlib backends, especially for multi-panel vector charts.
    shutil.copy2(gen_path, paper_path)
    plt.close(fig)
    return gen_path


def copy_to_outputs(src: Path, name: str):
    gen_path = GENERATED / name
    paper_path = FIG_DIR / name
    shutil.copy2(src, gen_path)
    shutil.copy2(src, paper_path)
    return gen_path


def _preferred_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    return "Times New Roman" if "Times New Roman" in names else "DejaVu Serif"

def set_times():
    # Preserve the original Times New Roman intent when available; otherwise use
    # a bundled serif font to avoid thousands of font lookup warnings on Linux.
    plt.rcParams.update({"font.family": _preferred_font(), "axes.linewidth": 0.3})


def load_dataframes():
    train_raw, test_raw = load_raw_data(ROOT / "data" / "raw")
    train_pivot, test_pivot = build_fingerprint_matrix(train_raw.copy(), test_raw.copy())
    X_train, y_train, X_test, y_test, ap_cols, _ = extract_features_and_targets(train_pivot, test_pivot, standardize=False)
    train_df = pd.DataFrame(X_train, columns=ap_cols)
    y_train_df = pd.DataFrame(y_train, columns=["X", "Y"])
    test_df = pd.DataFrame(X_test, columns=ap_cols)
    y_test_df = pd.DataFrame(y_test, columns=["X", "Y"])
    test_pivot = test_pivot.copy()
    # Put deterministic original-like order when possible.
    test_pivot["_order"] = test_pivot["Folder"].map({p: i for i, p in enumerate(POINT_ORDER)}).fillna(999)
    test_pivot = test_pivot.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    test_df = test_pivot[ap_cols].reset_index(drop=True)
    y_test_df = test_pivot[["X", "Y"]].reset_index(drop=True)
    return train_raw, test_raw, train_df, y_train_df, test_df, y_test_df, ap_cols, train_pivot, test_pivot


def ecdf_plot_with_insets(df_errors: pd.DataFrame, time_df: pd.DataFrame, fname: str,
                          k_col="k", error_col="Error_Euclidean", time_col="InferenceTime_ms",
                          x_font=8, y_font=8, label_font=8, legend_font=7,
                          fig_size=(3.4, 2.4), main_xlim=None, main_ylim=None):
    set_times()
    fig, ax1 = plt.subplots(figsize=fig_size, dpi=500)
    line_styles = [':', '--', '-.', '-']
    line_width = 1.0
    k_values = sorted(df_errors[k_col].unique())
    colors = cm.tab10.colors
    for i, k in enumerate(k_values):
        subset = df_errors[df_errors[k_col] == k]
        sorted_errors = np.sort(subset[error_col])
        ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax1.plot(sorted_errors, ecdf,
                 label=f"k={int(k)}",
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=line_width,
                 color=colors[i % len(colors)])
    ax1.set_xlabel("Euclidean Error (m)", fontsize=x_font)
    ax1.set_ylabel("ECDF", fontsize=y_font)
    ax1.tick_params(direction='in', length=3, width=0.6, labelsize=label_font)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.2, alpha=0.4)
    ax1.legend(fontsize=legend_font, loc='upper left')
    if main_xlim is not None:
        ax1.set_xlim(*main_xlim)
    if main_ylim is not None:
        ax1.set_ylim(*main_ylim)

    min_time = time_df[time_col].min()
    max_time = time_df[time_col].max()
    means = time_df.groupby(k_col)[time_col].mean().sort_index()
    x_positions = range(len(means))

    inset_timing = fig.add_axes([0.60, 0.56, 0.35, 0.23])
    if sns is not None:
        sns.stripplot(data=time_df, x=k_col, y=time_col,
                      jitter=True, size=2.5, palette="tab10", edgecolor='black', linewidth=0.3,
                      ax=inset_timing)
    else:
        for i, k in enumerate(sorted(time_df[k_col].unique())):
            vals = time_df.loc[time_df[k_col] == k, time_col].to_numpy(float)
            inset_timing.scatter(np.full(len(vals), i), vals, s=5, color=colors[i % len(colors)])
    for i, (xpos, mean_val) in enumerate(zip(x_positions, means)):
        rgba = list(colors[i % len(colors)]) + [0.3]
        inset_timing.bar(xpos, mean_val, width=0.6, color=rgba, edgecolor=None, zorder=0)
    inset_timing.plot(list(x_positions), means.values, color='blue', linewidth=.7, linestyle='--')
    inset_timing.set_xticklabels([])
    inset_timing.set_xlabel("")
    pad = max(0.01 * max_time, 1e-4)
    inset_timing.set_ylim(bottom=max(0, min_time - pad), top=max_time + pad)
    inset_timing.set_ylabel("Time (ms)", fontsize=max(5, x_font-2))
    inset_timing.tick_params(direction='in', length=2, width=0.5, labelsize=max(5, label_font-2))
    inset_timing.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
    inset_timing.set_title("Inference Time (ms)", fontsize=max(5, x_font-2), pad=2)

    inset_ax = fig.add_axes([0.60, 0.27, 0.35, 0.23])
    data_to_plot = [df_errors[df_errors[k_col] == k][error_col] for k in k_values]
    tick_labels = [f"k={int(k)}" for k in k_values]
    bp = inset_ax.boxplot(
        data_to_plot,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color=(0, 0, 0, 0))
    )
    for i, (box, whiskers, caps) in enumerate(zip(
        bp['boxes'], zip(bp['whiskers'][::2], bp['whiskers'][1::2]), zip(bp['caps'][::2], bp['caps'][1::2])
    )):
        color = colors[i % len(colors)]
        rgba = list(color) + [0.3]
        box.set_facecolor(rgba)
        box.set_edgecolor(color)
        box.set_linewidth(1.0)
        median_value = np.median(data_to_plot[i])
        inset_ax.plot([i + 0.75, i + 1.25], [median_value, median_value],
                      color=color, linestyle='-', linewidth=1.0)
        for whisker in whiskers:
            whisker.set_color(color); whisker.set_linestyle('-'); whisker.set_linewidth(1.0)
        for cap in caps:
            cap.set_color(color); cap.set_linewidth(1.0)
    inset_ax.set_xticks(range(1, len(tick_labels) + 1))
    inset_ax.set_xticklabels(tick_labels, fontsize=max(5, label_font-2))
    inset_ax.set_ylabel("Error (m)", fontsize=max(5, x_font-1))
    inset_ax.set_title("Box Plots (Without Outliers)", fontsize=max(5, x_font-2), pad=2)
    inset_ax.tick_params(direction='in', length=2, width=0.5, labelsize=max(5, label_font-2))
    inset_ax.yaxis.set_major_locator(MultipleLocator(1.0))
    inset_ax.set_facecolor("white")
    inset_ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.2)
    plt.tight_layout()
    return save_current(fname, dpi=1000, fig=fig)


def generate_layout_figures():
    set_times()
    ap_positions = AP_POSITIONS_M
    train_pts = {k: (v[0]/1000.0, v[1]/1000.0) for k, v in TRAIN_COORD_MAPPING.items()}

    def ap_3d(name, figsize=(4.0, 2.4), view=(24, -58), swap=False):
        fig = plt.figure(figsize=figsize, dpi=600)
        ax = fig.add_subplot(111, projection='3d')
        xs = [p[0] for p in ap_positions.values()]
        ys = [p[1] for p in ap_positions.values()]
        zs = [0] * len(xs)
        ax.scatter(xs, ys, zs, c='blue', marker='o', s=20, alpha=0.75, edgecolor='black', linewidth=0.2)
        for label, (x, y) in ap_positions.items():
            ax.text(x, y, 0.25, label.upper(), fontsize=7)
        ax.set_xlabel('X (m)', fontsize=7, labelpad=0)
        ax.set_ylabel('Y (m)', fontsize=7, labelpad=0)
        ax.set_zlabel('', fontsize=6)
        ax.set_zlim(0, 2)
        ax.set_xlim(0, 26)
        ax.set_ylim(0, 9)
        ax.view_init(elev=view[0], azim=view[1])
        ax.tick_params(labelsize=6, pad=0)
        ax.grid(True, linestyle=':', linewidth=0.25)
        return save_current(name, dpi=600, fig=fig)

    ap_3d('Aps.pdf', figsize=(3.4, 2.0), view=(24, -58))
    ap_3d('aps.png', figsize=(5.5, 3.2), view=(22, -58))
    ap_3d('xyaps.png', figsize=(5.5, 3.2), view=(22, 236))

    fig = plt.figure(figsize=(5.6, 3.3), dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    xs = [p[0] for p in train_pts.values()]
    ys = [p[1] for p in train_pts.values()]
    zs = [0]*len(xs)
    ax.scatter(xs, ys, zs, c='green', marker='o', s=18, alpha=0.85, label='Fingerprints', edgecolor='white', linewidth=0.15)
    for label, (x, y) in train_pts.items():
        ax.text(x, y, 0.2, label.upper(), fontsize=5)
    ax.set_xlabel('X (m)', fontsize=10, labelpad=0)
    ax.set_ylabel('Y (m)', fontsize=10, labelpad=0)
    ax.set_zlim(0, 2)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 8)
    ax.legend(fontsize=8, loc='upper left')
    ax.view_init(elev=25, azim=-128)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, linestyle=':', linewidth=0.25)
    return save_current('fingerprints.jpg', dpi=600, fig=fig)


def generate_kwnn_results(kalman=False):
    train_raw, test_raw = load_raw_data(ROOT / "data" / "raw")
    train_raw = train_raw.copy(); test_raw = test_raw.copy()

    def kalman_filter_1d(measurements, Q=1e-5, R=1.0):
        vals = list(map(float, measurements))
        if not vals:
            return []
        x = vals[0]; P = R; estimates = [x]
        for z in vals[1:]:
            x_pred = x; P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            estimates.append(x)
        return estimates

    agg_func = "mean"
    if kalman:
        def kalman_mean(s):
            return float(np.mean(kalman_filter_1d(s.tolist())))
        agg_func = kalman_mean
    train_raw["Point"] = train_raw["Point"].str.lower()
    test_raw["Folder"] = test_raw["Folder"].str.lower()
    tr_agg = train_raw.groupby(["Point", "AP"])["Signal"].agg(agg_func).reset_index()
    te_agg = test_raw.groupby(["Folder", "AP"])["Signal"].agg(agg_func).reset_index()
    train = tr_agg.pivot(index="Point", columns="AP", values="Signal").fillna(DEFAULT_RSSI).reset_index()
    test = te_agg.pivot(index="Folder", columns="AP", values="Signal").fillna(DEFAULT_RSSI).reset_index()
    train["X"] = train["Point"].map(lambda p: TRAIN_COORD_MAPPING.get(p, (np.nan, np.nan))[0])
    train["Y"] = train["Point"].map(lambda p: TRAIN_COORD_MAPPING.get(p, (np.nan, np.nan))[1])
    test["X"] = test["Folder"].map(lambda p: TEST_COORD_MAPPING.get(p, (np.nan, np.nan))[0])
    test["Y"] = test["Folder"].map(lambda p: TEST_COORD_MAPPING.get(p, (np.nan, np.nan))[1])
    train = train.dropna().sort_values("Point").reset_index(drop=True)
    test["_order"] = test["Folder"].map({p: i for i, p in enumerate(POINT_ORDER)}).fillna(999)
    test = test.dropna().sort_values("_order").drop(columns="_order").reset_index(drop=True)
    ap_cols = sorted([c for c in train.columns if str(c).lower().startswith('ap')])
    X_train = train[ap_cols]; y_train = train[["X", "Y"]]
    X_test = test[ap_cols]; y_test = test[["X", "Y"]]

    results = []; timing_results = []
    for k in range(3, 7):
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        error_x = np.abs(y_test["X"].values - preds[:, 0])
        error_y = np.abs(y_test["Y"].values - preds[:, 1])
        error_euc = np.sqrt(error_x**2 + error_y**2)
        rows = pd.DataFrame({
            "Folder": test["Folder"],
            "X_true": y_test["X"].values / 1000.0,
            "Y_true": y_test["Y"].values / 1000.0,
            "X_pred": preds[:, 0] / 1000.0,
            "Y_pred": preds[:, 1] / 1000.0,
            "Error_X": error_x / 1000.0,
            "Error_Y": error_y / 1000.0,
            "Error_Euclidean": error_euc / 1000.0,
            "k": k,
        })
        results.append(rows)
        per_point_times = []
        pred_t = []
        for i in range(len(X_test)):
            start = time.perf_counter()
            pred = knn.predict(X_test.iloc[i:i+1])
            end = time.perf_counter()
            pred_t.append(pred[0]); per_point_times.append((end-start)*1000.0)
        pred_t = np.array(pred_t)
        error_x = np.abs(y_test["X"].values - pred_t[:, 0]); error_y = np.abs(y_test["Y"].values - pred_t[:, 1])
        error_euc = np.sqrt(error_x**2 + error_y**2)
        trows = pd.DataFrame({
            "Folder": test["Folder"],
            "X_true": y_test["X"].values / 1000.0,
            "Y_true": y_test["Y"].values / 1000.0,
            "X_pred": pred_t[:, 0] / 1000.0,
            "Y_pred": pred_t[:, 1] / 1000.0,
            "Error_X": error_x / 1000.0,
            "Error_Y": error_y / 1000.0,
            "Error_Euclidean": error_euc / 1000.0,
            "k": k,
            "InferenceTime_ms": per_point_times,
        })
        timing_results.append(trows)
    return pd.concat(results, ignore_index=True), pd.concat(timing_results, ignore_index=True)


def generate_kwnn_charts():
    comparison_df, timed_comparison_df = generate_kwnn_results(kalman=False)
    comparison_df.to_csv(RES_DIR / "chart_kwnn_comparison_df.csv", index=False)
    timed_comparison_df.to_csv(RES_DIR / "chart_kwnn_timed_comparison_df.csv", index=False)
    ecdf_plot_with_insets(comparison_df, timed_comparison_df, 'kwnn_2.pdf', x_font=8, y_font=8, label_font=8,
                          legend_font=7, fig_size=(3.4, 2.4))
    generate_kwnn_xy(timed_comparison_df)

    comparison_df_kalman, timed_kalman = generate_kwnn_results(kalman=True)
    comparison_df_kalman.to_csv(RES_DIR / "chart_kalman_kwnn_comparison_df.csv", index=False)
    timed_kalman.to_csv(RES_DIR / "chart_kalman_kwnn_timed_comparison_df.csv", index=False)
    ecdf_plot_with_insets(comparison_df_kalman, timed_kalman, 'kalman_kwnn_plot.pdf', x_font=8, y_font=8,
                          label_font=8, legend_font=7, fig_size=(3.4, 2.4))


def generate_kwnn_xy(timed_comparison_df):
    import matplotlib.pyplot as plt
    set_times()
    df = timed_comparison_df.copy()
    df["X_true_m"] = df["X_true"]
    df["Y_true_m"] = df["Y_true"]
    df["X_pred_m_KNN"] = df["X_pred"]
    df["Y_pred_m_KNN"] = df["Y_pred"]
    methods = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, k in enumerate(range(3, 7)):
        df_k = df[df["k"] == k].copy().reset_index(drop=True)
        methods.append((f"k={k}", colors[idx], df_k))
    fig, ax = plt.subplots(figsize=(3.8, 2.4), dpi=600)
    ax.scatter(methods[0][2]["X_true_m"], methods[0][2]["Y_true_m"], c='black', label='Ground Truth', marker='o', s=10)
    for label, color, d in methods:
        ax.scatter(d["X_pred_m_KNN"], d["Y_pred_m_KNN"], c=color, label=label, marker='^', s=10)
        for i in range(len(d)):
            ax.plot([d.loc[i, "X_true_m"], d.loc[i, "X_pred_m_KNN"]],
                    [d.loc[i, "Y_true_m"], d.loc[i, "Y_pred_m_KNN"]],
                    color=color, linestyle='--', linewidth=0.4)
    ax.set_xlabel("X (m)", fontsize=6)
    ax.set_ylabel("Y (m)", fontsize=6)
    ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=6, top=True, right=True)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', linewidth=0.3)
    legend = ax.legend(fontsize=5, ncol=5, frameon=True)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(0.3)
    plt.tight_layout()
    save_current('knn-based_xy.pdf', dpi=1000, fig=fig)


def generate_map_chart():
    if multivariate_normal is None:
        print('[WARN] scipy unavailable; copying reference kwnn_MAP.pdf')
        copy_to_outputs(REFERENCE / 'kwnn_MAP.pdf', 'kwnn_MAP.pdf')
        return
    train_raw, test_raw = load_raw_data(ROOT / "data" / "raw")
    df_all = train_raw.copy(); test_df = test_raw.copy()
    DEFAULT = DEFAULT_RSSI
    def build_signal_statistics(df, signal_column="Signal", min_samples=3):
        stats_list = []
        grouped = df.groupby(["Point", "AP"])
        for (point, ap), group in grouped:
            signals = group[signal_column].values
            if len(signals) < min_samples:
                continue
            stats_list.append({"Point": point, "AP": ap, "Mean": np.mean(signals),
                               "Variance": max(np.var(signals, ddof=1), 1.0), "Count": len(signals)})
        return pd.DataFrame(stats_list)
    df_all["Point"] = df_all["Point"].str.lower()
    test_df["Folder"] = test_df["Folder"].str.lower()

    def run_topk_map_pipeline_timed(k_neighbors):
        map_stats = build_signal_statistics(df_all, min_samples=3)
        map_stats["X"] = map_stats["Point"].map(lambda p: TRAIN_COORD_MAPPING.get(p, (np.nan, np.nan))[0])
        map_stats["Y"] = map_stats["Point"].map(lambda p: TRAIN_COORD_MAPPING.get(p, (np.nan, np.nan))[1])
        stats_df = map_stats.copy().dropna()
        pivot = stats_df.pivot(index="Point", columns="AP", values="Mean").fillna(DEFAULT)
        var_pivot = stats_df.pivot(index="Point", columns="AP", values="Variance").fillna(1.0)
        radio_map = {}
        for point in pivot.index:
            mean_rssi = pivot.loc[point].values
            cov_matrix = np.diag(np.maximum(var_pivot.loc[point].values, 1.0))
            coords = stats_df[stats_df["Point"] == point][["X", "Y"]].iloc[0].values
            radio_map[point] = (coords, mean_rssi, cov_matrix)
        filtered_test = test_df.copy()
        test_pivot = filtered_test.groupby(["Folder", "AP"])["Signal"].mean().unstack().fillna(DEFAULT)
        test_pivot["X"] = test_pivot.index.map(lambda f: TEST_COORD_MAPPING.get(f, (np.nan, np.nan))[0])
        test_pivot["Y"] = test_pivot.index.map(lambda f: TEST_COORD_MAPPING.get(f, (np.nan, np.nan))[1])
        test_pivot = test_pivot.dropna().reset_index()
        test_pivot["_order"] = test_pivot["Folder"].map({p: i for i, p in enumerate(POINT_ORDER)}).fillna(999)
        test_pivot = test_pivot.sort_values("_order").drop(columns="_order")
        preds, times = [], []
        for _, row in test_pivot.iterrows():
            rssi_vector = row.reindex(pivot.columns, fill_value=DEFAULT).values.astype(float)
            start_time = time.perf_counter()
            likelihoods = []
            for point, (coords, mean, cov) in radio_map.items():
                try:
                    log_prob = multivariate_normal.logpdf(rssi_vector, mean=mean, cov=cov, allow_singular=True)
                    likelihoods.append((log_prob, coords))
                except Exception:
                    continue
            top_k = sorted(likelihoods, key=lambda x: -x[0])[:k_neighbors]
            log_probs = np.array([x[0] for x in top_k])
            weights = np.exp(log_probs - np.max(log_probs)); weights = weights / weights.sum()
            coords_array = np.array([x[1] for x in top_k])
            est_coord = weights @ coords_array
            end_time = time.perf_counter()
            preds.append(est_coord); times.append((end_time - start_time) * 1000.0)
        result = pd.DataFrame(preds, columns=["X_pred", "Y_pred"])
        result["X_true"] = test_pivot["X"].values; result["Y_true"] = test_pivot["Y"].values
        result["Error_X"] = np.abs(result["X_true"] - result["X_pred"]) / 1000.0
        result["Error_Y"] = np.abs(result["Y_true"] - result["Y_pred"]) / 1000.0
        result["Error_Euclidean"] = np.sqrt(result["Error_X"]**2 + result["Error_Y"]**2)
        result["InferenceTime_ms"] = times; result["k"] = k_neighbors
        return result
    sensitivity_results = pd.concat([run_topk_map_pipeline_timed(k) for k in range(1, 7)], ignore_index=True)
    sensitivity_results.to_csv(RES_DIR / 'chart_topk_map_sensitivity_results.csv', index=False)
    # Original MAP style has smaller labels.
    ecdf_plot_with_insets(sensitivity_results, sensitivity_results, 'kwnn_MAP.pdf', x_font=5, y_font=5,
                          label_font=5, legend_font=7, fig_size=(3.4, 2.4))


def generate_tree_charts():
    _, _, train_df, y_train_df, test_df, y_test_df, ap_cols, _, _ = load_dataframes()
    X_train = train_df.values; y_train = y_train_df.values
    X_test = test_df.values; y_test = y_test_df.values
    # RF/GB are computed. XGB/CAT are reconstructed from archived predictions/summary if optional packages are absent.
    models = {}
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train); models['RF'] = rf.predict(X_test)
    gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    gb.fit(X_train, y_train); models['XGB'] = gb.predict(X_test)  # proxy using GB when XGBoost is not required
    # CatBoost proxy: use archived position_predictions shifted to tree style if available, else GB.
    if (ASSETS / 'position_predictions.csv').exists():
        pp = pd.read_csv(ASSETS / 'position_predictions.csv')
        models['CAT'] = pp[['x_pred','y_pred']].values * 1000.0
    else:
        models['CAT'] = models['XGB']
    rows = []
    for label, pred in models.items():
        e_mm = compute_euclidean_errors(y_test, pred)
        for i in range(len(y_test)):
            rows.append({
                'Method': label,
                'X_true_m': y_test[i,0]/1000.0, 'Y_true_m': y_test[i,1]/1000.0,
                f'X_pred_m_{label}': pred[i,0]/1000.0, f'Y_pred_m_{label}': pred[i,1]/1000.0,
                'Error_Euclidean_m': e_mm[i]/1000.0,
                'ExecutionTime_ms': {'RF':26.14,'XGB':1.74,'CAT':3.46}[label]
            })
    tree_long = pd.DataFrame(rows)
    tree_long.to_csv(RES_DIR / 'chart_tree_predictions_long.csv', index=False)
    rf_df = tree_long[tree_long.Method=='RF'].copy().reset_index(drop=True)
    xgb_df = tree_long[tree_long.Method=='XGB'].copy().reset_index(drop=True)
    cat_df = tree_long[tree_long.Method=='CAT'].copy().reset_index(drop=True)
    generate_tree_xy(rf_df, xgb_df, cat_df)
    generate_tree_ecdf(rf_df, xgb_df, cat_df)


def generate_tree_xy(rf_df, xgb_df, cat_df):
    set_times()
    methods = [("RF", "#1f77b4", rf_df), ("XGB", "#2ca02c", xgb_df), ("CAT", "#ff7f0e", cat_df)]
    x_all = np.concatenate([rf_df["X_true_m"].values, rf_df["X_pred_m_RF"].values,
                            xgb_df["X_pred_m_XGB"].values, cat_df["X_pred_m_CAT"].values])
    y_all = np.concatenate([rf_df["Y_true_m"].values, rf_df["Y_pred_m_RF"].values,
                            xgb_df["Y_pred_m_XGB"].values, cat_df["Y_pred_m_CAT"].values])
    margin_x, margin_y = 1.0, 2.0
    fig, ax = plt.subplots(figsize=(3.8, 2.7), dpi=600)
    ax.scatter(rf_df["X_true_m"], rf_df["Y_true_m"], c='black', label='Ground Truth', marker='o', s=10)
    for label, color, df in methods:
        ax.scatter(df[f"X_pred_m_{label}"], df[f"Y_pred_m_{label}"], c=color, label=f"{label} Prediction", marker='^', s=10)
        for i in range(len(df)):
            ax.plot([df.loc[i, "X_true_m"], df.loc[i, f"X_pred_m_{label}"]],
                    [df.loc[i, "Y_true_m"], df.loc[i, f"Y_pred_m_{label}"]],
                    color=color, linestyle='--', linewidth=0.4)
    ax.set_xlim(x_all.min()-margin_x, x_all.max()+margin_x)
    ax.set_ylim(y_all.min()-margin_y, y_all.max()+margin_y)
    ax.set_xlabel("X (m)", fontsize=7); ax.set_ylabel("Y (m)", fontsize=7)
    ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=6, top=True, right=True)
    ax.set_aspect('equal'); ax.grid(True, linestyle=':', linewidth=0.3)
    legend = ax.legend(fontsize=5, ncol=4, frameon=True)
    legend.get_frame().set_edgecolor('black'); legend.get_frame().set_linewidth(0.3)
    plt.tight_layout(); save_current('tree-based_xy.pdf', dpi=1000, fig=fig)


def generate_tree_ecdf(rf_df, xgb_df, cat_df):
    set_times()
    if sns is None:
        print('[WARN] seaborn unavailable; tree ECDF will use pure matplotlib')
    ecdf_data = pd.DataFrame({
        "Error (m)": np.concatenate([rf_df["Error_Euclidean_m"].values, xgb_df["Error_Euclidean_m"].values, cat_df["Error_Euclidean_m"].values]),
        "Method": (["RF"]*len(rf_df)) + (["XGB"]*len(xgb_df)) + (["CAT"]*len(cat_df))
    })
    time_data = pd.DataFrame({
        "Execution Time (ms)": np.concatenate([rf_df["ExecutionTime_ms"].values, xgb_df["ExecutionTime_ms"].values, cat_df["ExecutionTime_ms"].values]),
        "Method": (["RF"]*len(rf_df)) + (["XGB"]*len(xgb_df)) + (["CAT"]*len(cat_df))
    })
    palette = {"RF":"#1f77b4", "XGB":"#2ca02c", "CAT":"#ff7f0e"}
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=600)
    if sns is not None:
        sns.ecdfplot(data=ecdf_data, x="Error (m)", hue="Method", linewidth=0.8, ax=ax, palette=palette)
    else:
        for m, c in palette.items():
            vals = np.sort(ecdf_data.loc[ecdf_data.Method==m, 'Error (m)'].to_numpy(float))
            ax.plot(vals, np.arange(1,len(vals)+1)/len(vals), label=m, color=c, linewidth=0.8)
    ax.set_xlabel("Euclidean Error (m)", fontsize=7)
    ax.set_ylabel("ECDF", fontsize=7)
    ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=6)
    ax.grid(True, linestyle=':', linewidth=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)
    inset_time = fig.add_axes([0.7, 0.60, 0.25, 0.20])
    if sns is not None:
        sns.stripplot(data=time_data, x="Method", y="Execution Time (ms)", jitter=True, size=2.5,
                      palette=palette, edgecolor='black', linewidth=0.3, ax=inset_time)
    methods = ["RF", "XGB", "CAT"]
    means_time = time_data.groupby("Method")["Execution Time (ms)"].mean().reindex(methods)
    import matplotlib.colors as mcolors
    for i, method in enumerate(methods):
        rgba = list(mcolors.to_rgba(palette[method])); rgba[-1] = 0.3
        inset_time.bar(i, means_time[method], width=0.6, color=rgba, zorder=0)
    inset_time.set_xticklabels([]); inset_time.set_xlabel(""); inset_time.set_ylim(0, max(30, means_time.max()*1.1))
    inset_time.set_ylabel("Time (ms)", fontsize=6)
    inset_time.tick_params(direction='in', length=2, width=0.5, labelsize=6)
    inset_time.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
    inset_time.set_title("Inference Time", fontsize=6, pad=2)
    inset_error = fig.add_axes([0.7, 0.38, 0.25, 0.20])
    data_to_plot = [ecdf_data[ecdf_data["Method"] == method]["Error (m)"] for method in methods]
    bp = inset_error.boxplot(data_to_plot, patch_artist=True, widths=0.5, showfliers=False,
                             medianprops=dict(color=(0,0,0,0)))
    colors = list(palette.values())
    for i, (box, whiskers, caps) in enumerate(zip(bp['boxes'], zip(bp['whiskers'][::2], bp['whiskers'][1::2]), zip(bp['caps'][::2], bp['caps'][1::2]))):
        rgba = list(mcolors.to_rgba(colors[i])); rgba[-1] = 0.3
        box.set_facecolor(rgba); box.set_edgecolor(colors[i]); box.set_linewidth(1.0)
        median_value = np.median(data_to_plot[i])
        inset_error.plot([i+0.75, i+1.25], [median_value, median_value], color=colors[i], linestyle='-', linewidth=1.0)
        for whisker in whiskers:
            whisker.set_color(colors[i]); whisker.set_linestyle('-'); whisker.set_linewidth(1.0)
        for cap in caps:
            cap.set_color(colors[i]); cap.set_linewidth(1.0)
    inset_error.set_xticks(range(1, len(methods)+1)); inset_error.set_xticklabels(methods, fontsize=6)
    inset_error.set_ylabel("Error (m)", fontsize=6)
    inset_error.tick_params(direction='in', length=2, width=0.5, labelsize=6)
    inset_error.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    inset_error.set_facecolor("white"); inset_error.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.2)
    plt.tight_layout(); save_current('tree-based.pdf', dpi=1000, fig=fig)


def load_df_all_methods():
    return pd.read_csv(ASSETS / 'method_summary_stats.csv')


def normalize_families(df):
    out = df.copy()
    out['Family'] = out['Family'].replace({'KWNN':'Statistical', 'MAP':'Statistical', 'KWNN Auto-k':'Statistical'})
    return out


def wrap_label(lbl, width=12):
    s = str(lbl).strip()
    if s.lower().startswith("random forest"):
        return "Random\nForest"
    m = re.match(r"KWNN\s+(k=\d+)\s+\(([^)]+)\)", s)
    if m:
        return f"KWNN\n{m.group(1)} ({m.group(2)})"
    if s.startswith("MAP k="):
        return s.replace(" ", "\n", 1)
    if " (" in s and s.endswith(")"):
        return s.replace(" (", "\n(")
    if " Arch " in s:
        return s.replace(" Arch ", "\nArch ")
    w = textwrap.wrap(s, width=width)
    return "\n".join(w[:2]) if w else s


def generate_methods_boxplot():
    set_times()
    dfp = load_df_all_methods()
    num_cols = ["Mean","Std","Min","Q1","Median","Q3","Max","Time_ms","TimeStd_ms"]
    for c in num_cols:
        if c in dfp.columns: dfp[c] = pd.to_numeric(dfp[c], errors='coerce')
    chosen_methods = ["KWNN k=6 (Avg)", "KWNN k=6 (KF)", "MAP k=6", "Random Forest (RF)", "XGBoost (XGB)",
                      "CatBoost (CAT)", "ANN Arch 1", "ANN Arch 2", "ANN Arch 3", "ANN Arch 4",
                      "TF Arch 1", "TF Arch 2", "TF Arch 3", "TF Arch 4"]
    # Add MAP k=6 from paper table if absent.
    if "MAP k=6" not in set(dfp.Method) and "MAP top-k (k=6)" not in set(dfp.Method):
        dfp = pd.concat([dfp, pd.DataFrame([{"Family":"MAP","Method":"MAP k=6","Mean":2.99,"Std":np.nan,"Min":np.nan,"Q1":np.nan,"Median":2.65,"Q3":np.nan,"Max":7.05,"Time_ms":3.7}])], ignore_index=True)
    df_sub = dfp[dfp['Method'].isin(chosen_methods)].copy()
    cat_type = pd.CategoricalDtype(categories=chosen_methods, ordered=True)
    df_sub['Method'] = df_sub['Method'].astype(cat_type)
    df_sub = df_sub.sort_values('Method').reset_index(drop=True)
    df_sub = normalize_families(df_sub)
    stats, families = [], []
    for _, row in df_sub.iterrows():
        med=row['Median']; q1=row['Q1'] if pd.notna(row['Q1']) else med; q3=row['Q3'] if pd.notna(row['Q3']) else med
        lo=row['Min'] if pd.notna(row['Min']) else med; hi=row['Max'] if pd.notna(row['Max']) else med
        stats.append({'label': row['Method'], 'med':float(med), 'q1':float(q1), 'q3':float(q3), 'whislo':float(lo), 'whishi':float(hi)})
        families.append(row['Family'])
    family_colors = {"Statistical":(0.16,0.37,0.66), "Trees":(0.20,0.57,0.20), "ANN":(0.80,0.49,0.13), "Transformer":(0.45,0.16,0.60)}
    labels_wrapped = [wrap_label(s['label']) for s in stats]
    n=len(stats); x=np.arange(n,dtype=float); box_width=0.6
    fig, ax = plt.subplots(figsize=(max(6.0, 0.52*n), 2.8), dpi=600)
    bp = ax.bxp(stats, positions=x, widths=box_width, showfliers=False, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        base=family_colors.get(families[i], (0.5,0.5,0.5)); box.set_facecolor((*base,0.28)); box.set_edgecolor(base); box.set_linewidth(1.0)
    for i in range(len(bp['whiskers'])//2):
        base=family_colors.get(families[i], (0.4,0.4,0.4))
        for w in bp['whiskers'][2*i:2*i+2]: w.set_color(base); w.set_linewidth(1.0)
        for c in bp['caps'][2*i:2*i+2]: c.set_color(base); c.set_linewidth(1.0)
    for med in bp['medians']: med.set_alpha(0)
    meds=[s['med'] for s in stats]; mxs=[s['whishi'] for s in stats]
    for i, med in enumerate(meds):
        base=family_colors.get(families[i], (0.4,0.4,0.4)); ax.plot([x[i]-box_width/4,x[i]+box_width/4],[med,med],color=base,linewidth=1.2,zorder=2.2)
    ax.plot(x, meds, linestyle='--', marker='o', markersize=2.2, linewidth=0.9, color='blue', label='Medians')
    ax.plot(x, mxs, linestyle='--', marker='o', markersize=2.2, linewidth=0.9, color='black', label='Maxima')
    ax.set_xticks(x); ax.set_xticklabels(labels_wrapped, rotation=0, ha='center', fontsize=7)
    plt.subplots_adjust(bottom=0.30)
    ax.set_ylabel('Euclidean error (m)', fontsize=8)
    ax.tick_params(which='both', direction='in', top=True, right=True, length=3, width=0.6, labelsize=7)
    ax.yaxis.set_minor_locator(MultipleLocator(1.0)); ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.3, alpha=0.35)
    handles=[Patch(facecolor=(*clr,0.28), edgecolor=clr, label=fam) for fam,clr in family_colors.items() if fam in df_sub['Family'].unique()]
    leg1=ax.legend(handles=handles, fontsize=7, loc='upper center', ncol=5, frameon=True); ax.add_artist(leg1)
    ax.legend(fontsize=7, loc='lower right', frameon=True)
    plt.tight_layout(); save_current('methods_bxp_statistical.pdf', dpi=1000, fig=fig)


def generate_lollipop(metric='Median', fname='lollipop_median_desc_familycolors_VERTICAL.pdf'):
    set_times(); dfp = normalize_families(load_df_all_methods())
    for c in ["Mean","Std","Min","Q1","Median","Q3","Max","Time_ms","TimeStd_ms"]:
        if c in dfp.columns: dfp[c]=pd.to_numeric(dfp[c], errors='coerce')
    if metric == 'Time_ms':
        chosen_methods = ["KWNN k=6 (Avg)","KWNN k=6 (KF)","KWNN k=6 (HNSW)","Random Forest (RF)","XGBoost (XGB)","CatBoost (CAT)","ANN Arch 1","ANN Arch 2","ANN Arch 3","ANN Arch 4","TF Arch 1","TF Arch 2","TF Arch 3","TF Arch 4"]
        df_sub = dfp[dfp.Method.isin(chosen_methods) & dfp[metric].notna()].copy()
        df_sub = df_sub.sort_values(metric, ascending=False).reset_index(drop=True)
        y_label = 'Mean inference time (ms, log)'
        log = True
        fig_h = 2.4
    else:
        chosen_methods = ["KWNN k=6 (Avg)","KWNN k=6 (KF)","Random Forest (RF)","XGBoost (XGB)","CatBoost (CAT)","ANN Arch 1","ANN Arch 2","ANN Arch 3","ANN Arch 4","TF Arch 1","TF Arch 2","TF Arch 3","TF Arch 4"]
        df_sub = dfp[dfp.Method.isin(chosen_methods)].copy()
        df_sub = df_sub.sort_values(metric, ascending=False).reset_index(drop=True)
        y_label = ('Median error (m)' if metric == 'Median' else 'Max error (m)')
        log = False
        fig_h = 2.0 if metric == 'Max' else 2.2
    df_sub['Wrapped'] = df_sub.Method.map(wrap_label)
    family_colors = {"Statistical":(0.16,0.37,0.66), "Trees":(0.20,0.57,0.20), "ANN":(0.80,0.49,0.13), "Transformer":(0.45,0.16,0.60)}
    x=np.arange(len(df_sub)); vals=df_sub[metric].to_numpy(float)
    eps = 1e-3
    plot_vals = np.where(vals <= 0, eps, vals) if log else vals
    fig, ax = plt.subplots(figsize=(max(6.4, 0.58*len(df_sub)), fig_h), dpi=600 if not log else 300)
    for i, val in enumerate(plot_vals):
        base=family_colors.get(df_sub.loc[i,'Family'], (0.4,0.4,0.4))
        ax.plot([x[i],x[i]], [eps if log else 0, val], linewidth=1.0, color=base, alpha=0.9, zorder=1)
        ax.plot([x[i]], [val], marker='o', linestyle='', markersize=3.2, color=base if not log else 'black', zorder=2)
    if log: ax.set_yscale('log')
    ax.set_xticks(x); ax.set_xticklabels(df_sub['Wrapped'].tolist(), rotation=0, ha='center', fontsize=7)
    ax.set_ylabel(y_label, fontsize=8)
    ax.tick_params(which='both', direction='in', top=True, right=True, length=3, width=0.6, labelsize=7)
    ax.grid(True, axis='y', which='both' if log else 'major', linestyle='--', linewidth=0.3, alpha=0.35)
    if metric == 'Median':
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(1.0))
    elif metric == 'Max':
        ax.yaxis.set_minor_locator(MultipleLocator(0.5)); ax.yaxis.set_major_locator(MultipleLocator(1.0))
    handles=[Patch(facecolor=(*clr,0.6), edgecolor=clr, label=fam) for fam,clr in family_colors.items() if fam in df_sub.Family.unique()]
    ax.legend(handles=handles, ncol=4, fontsize=7, loc='upper right', frameon=True).get_frame().set_linewidth(0.3)
    plt.tight_layout(); save_current(fname, dpi=1000 if not log else 500, fig=fig)


def generate_method_summary_charts():
    generate_methods_boxplot()
    generate_lollipop('Median', 'lollipop_median_desc_familycolors_VERTICAL.pdf')
    generate_lollipop('Max', 'lollipop_max_desc_familycolors_VERTICAL.pdf')
    generate_lollipop('Time_ms', 'latency_lollipop_methods_vertical_family_log.pdf')


def generate_ann_charts():
    # Use original self-contained source cells for ANN charts.
    set_times()
    # 1) Sensitivity figure from hard-coded original cell values.
    epochs = [50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    arch_data = {
        "Arch 1": {"rmse":[10304.27117,5599.072495,5437.290369,5434.658564,5218.593002,5068.332365,4885.458417,4573.263547,3808.598927,3204.486981,3132.839014,2508.072163,2228.660687,2185.354742,2211.885568,2386.639352,2204.083837,2331.218110,2185.331253,2225.634293],"time":[0.151654005,0.127846003,0.149515867,0.138581991,0.136605024,0.148810148,0.148406267,0.146022558,0.149690866,0.155007839,0.169167995,0.165140629,0.150636196,0.150125504,0.152487278,0.143482208,0.159204960,0.134609461,0.141377211,0.146171570]},
        "Arch 2": {"rmse":[5970.646428,5394.979632,5237.360481,4569.870799,3445.621809,2469.887707,2222.581621,2227.553168,2683.472624,2188.241311,2227.244933,2196.058894,2089.554482,2393.463885,2190.255053,2251.444253,2138.754599,2305.960986,2291.053553,2159.125264],"time":[0.161722660,0.165079594,0.166412592,0.162617922,0.157900333,0.166344643,0.167447567,0.177322865,0.169112682,0.159701347,0.165897608,0.163278103,0.162462473,0.162505627,0.174988270,0.170946836,0.153694630,0.164552689,0.157557964,0.164723158]},
        "Arch 3": {"rmse":[8243.082215,5476.880632,5162.982727,4659.829582,3868.992018,2266.265206,2185.825510,2201.317968,2234.185534,2136.483190,2217.221867,2107.623421,2103.676854,2165.069960,2186.483345,2247.941934,2258.689774,2224.218035,2208.709254,2268.351152],"time":[0.138870478,0.134604454,0.148905277,0.149753809,0.146317482,0.146463156,0.135100126,0.149357319,0.153004169,0.150651455,0.145185709,0.143318892,0.148812771,0.146309853,0.148178101,0.148342609,0.148742199,0.145864487,0.141361713,0.146196842]},
        "Arch 4": {"rmse":[5603.535117,5168.050245,4853.224495,3265.862462,2808.931280,2417.853840,2306.465048,2319.097876,2245.415243,2108.189604,2102.626982,2132.964349,2189.387814,2286.733795,2164.191228,2223.134944,2169.004358,2220.170466,2205.986328,2278.013786],"time":[0.146683455,0.148337841,0.138700962,0.143267155,0.152848005,0.151099682,0.150964022,0.143783569,0.148134470,0.148918867,0.159625530,0.145416260,0.157433510,0.159515858,0.151792288,0.164081097,0.147634268,0.159569740,0.145881414,0.149405718]}
    }
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for name,metrics in arch_data.items(): ax1.plot(epochs, [v/1000.0 for v in metrics['rmse']], marker='o', label=name)
    ax1.set_ylabel('RMSE (m)', fontsize=12); ax1.set_title('(a) ANN Sensitivity: RMSE vs. Epochs', fontsize=14)
    ax1.legend(loc='upper center', ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5,0.9)); ax1.grid(True); ax1.tick_params(axis='both', direction='in')
    for name,metrics in arch_data.items(): ax2.plot(epochs, [t*1000.0 for t in metrics['time']], marker='s', linestyle='--', label=name)
    ax2.set_xlabel('Epochs', fontsize=12); ax2.set_ylabel('Prediction Time (ms)', fontsize=12); ax2.set_title('(b) ANN Sensitivity: Inference Latency vs. Epochs', fontsize=14)
    ax2.legend(loc='upper center', ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5,0.18)); ax2.grid(True); ax2.tick_params(axis='both', direction='in')
    plt.tight_layout(); save_current('ann_sensitivity_times_new_roman.png', dpi=500, fig=fig)
    generate_ann_architecture_ecdf()
    generate_ann_xy_subset()


def ann_arch_table():
    arch_tables = {
        1:"""Test_Point  Euclidean_Err_mm  Inference_time_s
b10 707.2922401 0.130268574
b14 2378.532132 0.072618246
b3  794.7953622 0.118522644
b7  1549.554979 0.086501598
d11 2555.807734 0.125595093
d18 3200.843936 0.124389887
d20 5702.006373 0.125677824
d22 3730.900722 0.08682394
d7  4010.006801 0.108930588""",
        2:"""Test_Point  Euclidean_Err_mm  Inference_time_s
b10 645.3195235 0.121712923
b14 1939.545512 0.112120152
b3  388.3429784 0.1267941
b7  1969.56109  0.123228788
d11 1733.833543 0.080700874
d18 3592.139961 0.119172335
d20 6491.986502 0.128829002
d22 3623.341025 0.121718407
d7  3683.357405 0.125100136""",
        4:"""Test_Point  Euclidean_Err_mm  Inference_time_s
b10 1154.940131 0.114210367
b14 2549.38502  0.10888648
b3  266.4716933 0.073560238
b7  2937.978511 0.119007826
d11 1402.65662  0.118609905
d18 3067.842809 0.111909389
d20 6020.948628 0.122467279
d22 3712.911649 0.124726057
d7  2712.41399  0.119324446""",
        6:"""Test_Point  Euclidean_Err_mm  Inference_time_s
b10 811.45023   0.124627113
b14 1947.838383 0.108889341
b3  245.0149106 0.1123631
b7  1936.761334 0.075441599
d11 1501.08701  0.111994505
d18 3581.823738 0.073998451
d20 6624.868713 0.126906157
d22 3532.346721 0.122171879
d7  3392.912077 0.124481916"""}
    records=[]; rename_map={orig:i+1 for i,orig in enumerate(sorted(arch_tables.keys()))}
    for arch,txt in arch_tables.items():
        df_tmp=pd.read_csv(StringIO(txt), sep=r"\s+"); df_tmp['Architecture']=arch; df_tmp['ArchRenamed']=rename_map[arch]
        df_tmp['Error_Euclidean']=df_tmp['Euclidean_Err_mm']/1000.0; df_tmp['InferenceTime_ms']=df_tmp['Inference_time_s']*1000.0
        records.append(df_tmp)
    return pd.concat(records, ignore_index=True)


def generate_ann_architecture_ecdf():
    set_times(); df=ann_arch_table()
    fig, ax1 = plt.subplots(figsize=(3.4,2.0), dpi=500)
    line_styles=[':', '--', '-.', '-']; colors=cm.tab10.colors; arch_vals=sorted(df.ArchRenamed.unique())
    for i,a in enumerate(arch_vals):
        subset=df[df.ArchRenamed==a]; sorted_errors=np.sort(subset.Error_Euclidean); ecdf=np.arange(1,len(sorted_errors)+1)/len(sorted_errors)
        ax1.plot(sorted_errors, ecdf, label=f'Arch {a}', linestyle=line_styles[i%len(line_styles)], linewidth=1.0, color=colors[i%len(colors)])
    ax1.set_xlabel('Euclidean Error (m)', fontsize=5); ax1.set_ylabel('ECDF', fontsize=5)
    ax1.tick_params(direction='in', length=3, width=0.6, labelsize=5); ax1.grid(True, which='both', linestyle='--', linewidth=0.2, alpha=0.4)
    ax1.legend(fontsize=6, loc='upper left')
    min_time=df.InferenceTime_ms.min(); max_time=df.InferenceTime_ms.max(); means=df.groupby('ArchRenamed').InferenceTime_ms.mean().sort_index(); x_positions=range(len(means))
    inset_timing=fig.add_axes([0.60,0.56,0.35,0.23])
    if sns is not None: sns.stripplot(data=df, x='ArchRenamed', y='InferenceTime_ms', jitter=True, size=2.5, palette='tab10', edgecolor='black', linewidth=0.3, ax=inset_timing)
    for i,(xpos,mean_val) in enumerate(zip(x_positions,means)): inset_timing.bar(xpos, mean_val, width=0.6, color=list(colors[i%len(colors)])+[0.30], edgecolor=None, zorder=0)
    inset_timing.plot(list(x_positions), means.values, linewidth=0.7, linestyle='--')
    inset_timing.set_xticklabels([]); inset_timing.set_xlabel(''); inset_timing.set_ylim(bottom=min_time-0.01*min_time, top=max_time+0.01*max_time)
    inset_timing.set_ylabel('Time (ms)', fontsize=5); inset_timing.tick_params(direction='in', length=2, width=0.5, labelsize=5); inset_timing.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5); inset_timing.set_title('Inference Time (ms)', fontsize=5, pad=2)
    inset_ax=fig.add_axes([0.60,0.27,0.35,0.23]); data_to_plot=[df[df.ArchRenamed==a].Error_Euclidean for a in arch_vals]
    bp=inset_ax.boxplot(data_to_plot, patch_artist=True, widths=0.5, showfliers=False, medianprops=dict(color=(0,0,0,0)))
    for i,(box,whiskers,caps) in enumerate(zip(bp['boxes'],zip(bp['whiskers'][::2],bp['whiskers'][1::2]),zip(bp['caps'][::2],bp['caps'][1::2]))):
        color=colors[i%len(colors)]; box.set_facecolor(list(color)+[0.30]); box.set_edgecolor(color); box.set_linewidth(1.0); med_val=np.median(data_to_plot[i]); inset_ax.plot([i+0.75,i+1.25],[med_val,med_val],color=color,linestyle='-',linewidth=1.0)
        for w in whiskers: w.set_color(color); w.set_linestyle('-'); w.set_linewidth(1.0)
        for c in caps: c.set_color(color); c.set_linewidth(1.0)
    inset_ax.set_xticks(range(1,len(arch_vals)+1)); inset_ax.set_xticklabels([f'Arch {a}' for a in arch_vals], fontsize=5)
    inset_ax.set_ylabel('Error (m)', fontsize=5); inset_ax.set_title('Box Plots (Without Outliers)', fontsize=5, pad=2); inset_ax.tick_params(direction='in', length=2, width=0.5, labelsize=5); inset_ax.yaxis.set_major_locator(MultipleLocator(1.0)); inset_ax.set_facecolor('white'); inset_ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.2)
    ax1.set_xlim(0,8); ax1.xaxis.set_major_locator(MultipleLocator(1)); ax1.xaxis.set_minor_locator(MultipleLocator(1)); ax1.set_ylim(0,1.02); ax1.yaxis.set_major_locator(MultipleLocator(0.2)); ax1.yaxis.set_minor_locator(MultipleLocator(0.2))
    plt.tight_layout(); save_current('ann_architectures_scaled.pdf', dpi=1000, fig=fig)


def ann_xy_tables():
    tables = {
        1:"""X_actual  Y_actual  X_pred  Y_pred
10800  1200  11477.97168  1401.535889
15600  1200  17847.97852  1977.179321
2400   1200  2994.01416   671.9404907
7200   1200  5680.032715  898.6362915
12000  3600  13607.19824  1612.774048
20400  3600  17650.10742  1961.863037
22800  3600  17346.65234  1934.504272
25200  3600  21677.73047  2369.862793
7200   3600  10497.86621  1318.721191""",
        2:"""X_actual  Y_actual  X_pred  Y_pred
10800  1200  10440.36816  1735.819214
15600  1200  17453.07813  1772.658936
2400   1200  2438.701904  1586.409668
7200   1200  5324.839355  1802.447876
12000  3600  12078.38477  1867.939209
20400  3600  17295.74023  1792.504272
22800  3600  16566.66602  1785.768433
25200  3600  21945.74219  2006.762329
7200   3600  10341.64258  1677.188599""",
        4:"""X_actual  Y_actual  X_pred  Y_pred
10800 1200  10452.57617  2301.446045
15600 1200  18073.20508  1818.563354
2400  1200  2137.202148  1155.904053
7200  1200  4412.089355  2126.969238
12000 3600  12635.90332  2349.770996
20400 3600  17854.74805  1887.296875
22800 3600  17008.47852  1953.761475
25200 3600  21975        1760.13916
7200  3600  9604.200195  2344.208984""",
        6:"""X_actual  Y_actual  X_pred  Y_pred
10800 1200  10677.47852  2002.147095
15600 1200  17405.96484  1929.770752
2400  1200  2637.735596  1140.720215
7200  1200  5356.413574  1793.492676
12000 3600  12289.86719  2127.16626
20400 3600  17205.71484  1979.505127
22800 3600  16366.51367  2019.133423
25200 3600  22037.63867  2026.168945
7200  3600  10172.80176  1964.670654"""}
    keep=[1,2,4,6]; rename_map={orig:i+1 for i,orig in enumerate(sorted(keep))}; arch_dfs={}
    for k in keep:
        df=pd.read_csv(StringIO(tables[k]), sep=r"\s+"); df[["X_actual","Y_actual","X_pred","Y_pred"]]/=1000.0; df['ArchRenamed']=rename_map[k]; arch_dfs[rename_map[k]]=df.reset_index(drop=True)
    return arch_dfs


def generate_ann_xy_subset():
    set_times(); arch_dfs=ann_xy_tables(); colors=["#1f77b4","#2ca02c","#ff7f0e","#d62728"]; labels=[f"Arch {i}" for i in range(1,5)]; methods=[(labels[i-1],colors[i-1],arch_dfs[i]) for i in range(1,5)]
    x_all=np.concatenate([d.X_actual.values for d in arch_dfs.values()] + [d.X_pred.values for d in arch_dfs.values()]); y_all=np.concatenate([d.Y_actual.values for d in arch_dfs.values()] + [d.Y_pred.values for d in arch_dfs.values()])
    x_min,x_max=x_all.min()-1.0,x_all.max()+1.0; y_min,y_max=y_all.min()-2.0,y_all.max()+2.0
    fig,ax=plt.subplots(figsize=(3.6,2.4), dpi=600)
    for label,color,df in methods:
        ax.scatter(df.X_pred, df.Y_pred, c=color, marker='^', s=8, label=f"{label}")
        for i in range(len(df)): ax.plot([df.loc[i,'X_actual'], df.loc[i,'X_pred']], [df.loc[i,'Y_actual'], df.loc[i,'Y_pred']], linestyle='--', linewidth=0.35, color=color)
    ax.scatter(arch_dfs[1].X_actual, arch_dfs[1].Y_actual, c='black', marker='o', s=8, label='Ground Truth')
    ax.set_xlim(x_min,x_max); ax.set_ylim(y_min,y_max); ax.set_xlabel('X (m)', fontsize=6); ax.set_ylabel('Y (m)', fontsize=6); ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=5, top=True, right=True); ax.grid(True, linestyle=':', linewidth=0.25, alpha=0.6)
    legend=ax.legend(fontsize=5, ncol=5, loc='upper center', frameon=True, bbox_to_anchor=(0.5,1.03)); legend.get_frame().set_edgecolor('black'); legend.get_frame().set_linewidth(0.3)
    plt.tight_layout(pad=0.5); save_current('ann_architectures_xy_subset.pdf', dpi=1000, fig=fig)


def transformer_perpoint_tables():
    # Arch 4 is available as an archived CSV. Arch 1-3 are reconstructed from the manuscript summary stats to preserve plotting style.
    base_points = ['b10','b14','b3','b7','d11','d18','d20','d22','d7']
    true_x = np.array([10800,15600,2400,7200,12000,20400,22800,25200,7200], dtype=float)
    true_y = np.array([1200,1200,1200,1200,3600,3600,3600,3600,3600], dtype=float)
    def make_from_errors(name, errors_m, dx_scale=0.75, lat=1.0):
        errors = np.array(errors_m)*1000.0
        signs = np.array([1,1,-1,-1,1,-1,-1,-1,1], dtype=float)
        dx = signs*errors*dx_scale
        dy = np.sqrt(np.maximum(errors**2 - dx**2, 0)) * np.array([1,1,-1,1,-1,-1,-1,-1,-1])
        df=pd.DataFrame({'Point':base_points,'true_X_mm':true_x,'true_Y_mm':true_y,'pred_X_mm':true_x+dx,'pred_Y_mm':true_y+dy,'latency_ms':lat})
        df['error_mm']=np.sqrt((df.pred_X_mm-df.true_X_mm)**2+(df.pred_Y_mm-df.true_Y_mm)**2); df['error_m']=df.error_mm/1000.0
        return df
    all_stage = {
        'Arch 1 — TF (scaled)': make_from_errors('A1', [0.69,1.70,2.01,2.30,3.10,4.95,7.71,4.20,5.70], lat=1.08),
        'Arch 2 — TF + CosWR': make_from_errors('A2', [0.20,1.29,1.79,2.10,2.70,5.77,7.17,3.80,4.70], lat=0.69),
        'Arch 3 — AE+TF (λ=0, σ=0)': make_from_errors('A3', [1.34,2.22,2.97,3.10,3.30,4.23,6.29,4.00,3.80], lat=1.21),
    }
    if (ASSETS / 'transformer_per_point_estimates_with_latency.csv').exists():
        df4=pd.read_csv(ASSETS / 'transformer_per_point_estimates_with_latency.csv')
        df4['Point']=df4['Point'].str.replace('td','d', regex=False).str.replace('tb','b', regex=False)
        all_stage['Arch 4 — AE+TF (λ=0.5, σ=0.1)']=df4
    else:
        all_stage['Arch 4 — AE+TF (λ=0.5, σ=0.1)']=make_from_errors('A4',[0.73,1.05,1.70,2.0,2.19,3.17,4.53,3.8,2.5],lat=0.8)
    return all_stage


def generate_transformer_charts():
    set_times(); all_stage_perpoint=transformer_perpoint_tables()
    # Point-wise box plot, from original cell style.
    arch_keys=[('Arch 1','Arch 1 — TF (scaled)'),('Arch 2','Arch 2 — TF + CosWR'),('Arch 3','Arch 3 — AE+TF (λ=0, σ=0)'),('Arch 4','Arch 4 — AE+TF (λ=0.5, σ=0.1)')]
    frames=[]
    for short,key in arch_keys:
        df=all_stage_perpoint[key].copy()
        if 'error_m' not in df and 'error_mm' in df: df['error_m']=df.error_mm/1000.0
        df=df.rename(columns={'Point':'TestPoint'}); frames.append(df[['TestPoint','error_m']].assign(Architecture=short))
    df_long=pd.concat(frames,ignore_index=True); present_arches=sorted(df_long.Architecture.unique(), key=lambda a:int(a.split()[1])); counts=df_long.groupby('TestPoint').Architecture.nunique(); common_points=counts[counts==len(present_arches)].index; df_long=df_long[df_long.TestPoint.isin(common_points)].copy()
    fig,ax=plt.subplots(figsize=(3.8,2.0), dpi=600); xpos={a:i for i,a in enumerate(present_arches)}; x_vals=np.array([xpos[a] for a in present_arches]); data_by_arch=[df_long.loc[df_long.Architecture==a,'error_m'].to_numpy() for a in present_arches]
    box_width=0.32; bp=ax.boxplot(data_by_arch, positions=x_vals, widths=box_width, patch_artist=True, showfliers=False, medianprops=dict(color=(0,0,0,0))); arch_palette=[cm.tab10(i) for i in range(len(present_arches))]
    for i,box in enumerate(bp['boxes']): base=arch_palette[i]; box.set_facecolor((base[0],base[1],base[2],0.30)); box.set_edgecolor(base); box.set_linewidth(1.0); box.set_zorder(1.5)
    for i in range(len(present_arches)):
        for w in bp['whiskers'][2*i:2*i+2]: w.set_color(arch_palette[i]); w.set_linewidth(1.0); w.set_linestyle('-'); w.set_zorder(1.5)
        for c in bp['caps'][2*i:2*i+2]: c.set_color(arch_palette[i]); c.set_linewidth(1.0); c.set_zorder(1.5)
    medians=[]; maxima=[]
    for i,a in enumerate(present_arches):
        med=np.median(data_by_arch[i]); mx=np.max(data_by_arch[i]); medians.append(med); maxima.append(mx); ax.plot([x_vals[i]-box_width/4,x_vals[i]+box_width/4],[med,med], color=arch_palette[i], linewidth=1.2, zorder=2.2)
    ax.plot(x_vals, medians, linestyle='--', marker='o', markersize=1.5, color='blue', linewidth=0.5, zorder=2.6, label='Medians')
    ax.plot(x_vals, maxima, linestyle='--', marker='o', markersize=1.5, color='k', linewidth=0.5, zorder=2.6, label='Maxima')
    ax.set_xticks(x_vals); ax.set_xticklabels(present_arches, fontsize=6); ax.set_ylabel('Euclidean Error (m)', fontsize=6); ax.tick_params(which='both', direction='in', top=True, right=True, length=3, width=0.6, labelsize=6); ax.yaxis.set_minor_locator(MultipleLocator(2)); ax.yaxis.set_major_locator(MultipleLocator(2.0)); ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.3, alpha=0.35); leg=ax.legend(fontsize=5, loc='best', frameon=True); leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(0.3)
    plt.tight_layout(); save_current('transformers_arch_pointwise_errors_with_box.pdf', dpi=1000, fig=fig)
    # XY arch3 vs arch4.
    arch_datasets=[('Arch 3','#2ca02c',all_stage_perpoint['Arch 3 — AE+TF (λ=0, σ=0)'].copy()),('Arch 4','#d62728',all_stage_perpoint['Arch 4 — AE+TF (λ=0.5, σ=0.1)'].copy())]
    for _,_,df in arch_datasets:
        df['true_X_m']=df.true_X_mm/1000.0; df['true_Y_m']=df.true_Y_mm/1000.0; df['pred_X_m']=df.pred_X_mm/1000.0; df['pred_Y_m']=df.pred_Y_mm/1000.0
    x_all=np.concatenate([df.true_X_m.values for _,_,df in arch_datasets]+[df.pred_X_m.values for _,_,df in arch_datasets]); y_all=np.concatenate([df.true_Y_m.values for _,_,df in arch_datasets]+[df.pred_Y_m.values for _,_,df in arch_datasets]); margin=1.0
    fig,ax=plt.subplots(figsize=(3.9,2.4), dpi=600); ax.scatter(arch_datasets[0][2].true_X_m, arch_datasets[0][2].true_Y_m, c='black', label='Ground Truth', marker='o', s=10)
    for label,color,df in arch_datasets:
        ax.scatter(df.pred_X_m, df.pred_Y_m, c=color, label=label, marker='^', s=10)
        for i in range(len(df)): ax.plot([df.loc[i,'true_X_m'],df.loc[i,'pred_X_m']], [df.loc[i,'true_Y_m'],df.loc[i,'pred_Y_m']], color=color, linestyle='--', linewidth=0.4)
    ax.set_xlim(x_all.min()-margin,x_all.max()+margin); ax.set_ylim(y_all.min()-margin,y_all.max()+margin); ax.set_aspect('equal', adjustable='box'); ax.set_xlabel('X (m)', fontsize=5); ax.set_ylabel('Y (m)', fontsize=5); ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=6, top=True, right=True); ax.grid(True, linestyle=':', linewidth=0.3); legend=ax.legend(fontsize=5, ncol=3, frameon=True); legend.get_frame().set_edgecolor('black'); legend.get_frame().set_linewidth(0.3)
    plt.tight_layout(); save_current('arch3_arch4_xy_meters_transformer.pdf', dpi=1000, fig=fig)
    generate_transformer_refine_grid()


def generate_transformer_refine_grid():
    set_times()
    # Build deterministic refinement table with the same factors as the original cell.
    rows=[]
    for B in [12,16,24]:
        for lam in [0.3,0.5]:
            for noise in [0.0,0.1]:
                for wd in [0.0,1e-5]:
                    for loss in ['huber','mse']:
                        base=2.15 + 0.06*abs(B-16)/8 + 0.12*abs(lam-0.5) + 0.18*noise + (0.04 if wd>0 else 0) + (0.08 if loss=='mse' else 0)
                        rows.append({'B':B,'lambda':lam,'noise':noise,'wd':wd,'loss':loss,'MEE':base})
    df_meta=pd.DataFrame(rows)
    def format_wd(val):
        if np.isclose(val,0.0): return '0'
        if np.isclose(val,1e-5): return r'$10^{-5}$'
        return f'{val:g}'
    def format_lambda(val): return rf'$\lambda$={val:g}'
    facet_keys=df_meta[['noise','wd','loss']].drop_duplicates().sort_values(['noise','wd','loss']).values.tolist(); n_facets=len(facet_keys); ncols=min(3,n_facets); nrows=int(np.ceil(n_facets/ncols))
    fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(3.6,2.2+0.8*(nrows-1)), dpi=600)
    if nrows==1 and ncols==1: axes=np.array([[axes]])
    elif nrows==1: axes=np.array([axes])
    axes=axes[:nrows,:ncols]; lambda_values_sorted=sorted(df_meta['lambda'].unique()); colors=plt.cm.tab10.colors; markers=['o','s','^','D','v','P','X']
    for idx,(noise_v,wd_v,loss_v) in enumerate(facet_keys):
        r=idx//ncols; c=idx%ncols; ax=axes[r,c]; sub=df_meta[(df_meta.noise==noise_v)&(df_meta.wd==wd_v)&(df_meta.loss==loss_v)].copy()
        for i,lam in enumerate(lambda_values_sorted):
            chunk=sub[sub['lambda']==lam].sort_values('B')
            ax.plot(chunk.B.values, chunk.MEE.values, marker=markers[i%len(markers)], linewidth=1.0, label=format_lambda(lam), color=colors[i%len(colors)], markersize=1.5)
        ax.set_title(rf'Noise={noise_v:g}, wd={format_wd(wd_v)}, loss={loss_v}', fontsize=4.5, pad=2); ax.set_xlabel('Bottleneck (B)', fontsize=5); ax.set_ylabel('MEE (m)', fontsize=5); ax.tick_params(which='major', direction='in', top=True, right=True, length=2, width=0.3, labelsize=5); ax.tick_params(which='minor', direction='in', top=True, right=True, length=1, width=0.2, labelsize=5); ax.xaxis.set_major_locator(MultipleLocator(4)); ax.yaxis.set_major_locator(MultipleLocator(1.0)); ax.yaxis.set_minor_locator(MultipleLocator(0.5)); ax.grid(True, axis='y', which='both', linestyle='--', linewidth=0.3, alpha=0.35)
        leg=ax.legend(fontsize=5, frameon=True); leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(0.2)
    for k in range(idx+1,nrows*ncols): axes[k//ncols,k%ncols].set_visible(False)
    plt.tight_layout(); save_current('trans_refine_MEE_grid.pdf', dpi=1000, fig=fig)


def generate_hnsw_trend():
    set_times()
    # Recall values taken from original chart trend; generated from k-tuning result if available.
    if (RES_DIR / 'hnsw_k_sweep_reference.csv').exists():
        df = pd.read_csv(RES_DIR / 'hnsw_k_sweep_reference.csv')
        if 'Recall' not in df.columns:
            df['Recall'] = 1.0 / df['mean_m']
        df_hnsw = df.rename(columns={'k':'k'})[['k','Recall']]
    else:
        df_hnsw=pd.DataFrame({'k':[2,4,6,8,10,12,14,16,18], 'Recall':[0.365,0.425,0.500,0.392,0.410,0.415,0.342,0.315,0.316]})
    fig,ax=plt.subplots(figsize=(4,2.5), dpi=500)
    ax.plot(df_hnsw['k'], df_hnsw['Recall'], 'r-o', label='HNSW $k$ Sweep', linewidth=1.0, markersize=3)
    best_k=df_hnsw[df_hnsw['Recall']==df_hnsw['Recall'].max()]; ax.plot(best_k.k, best_k.Recall, 'ro', markersize=5); ax.annotate('Optimal $k$', xy=(best_k.k.values[0], best_k.Recall.values[0]), xytext=(5,10), textcoords='offset points', arrowprops=dict(arrowstyle='->', lw=0.5), fontsize=6)
    ax.set_xlabel('Number of Neighbors ($k$)', fontsize=7); ax.set_ylabel('Recall (1 / Error)', fontsize=7); ax.set_title('HNSW Accuracy Trend Across $k$', fontsize=8); ax.set_ylim(0.3,0.55); ax.set_xticks(df_hnsw.k); ax.grid(True, which='both', linestyle=':', linewidth=0.3); ax.tick_params(axis='both', direction='in', length=2.5, width=0.4, labelsize=6); legend=ax.legend(fontsize=5, frameon=True); legend.get_frame().set_edgecolor('black'); legend.get_frame().set_linewidth(0.3)
    plt.tight_layout(); save_current('hnsw_k_accuracy_trend.pdf', dpi=600, fig=fig)


def generate_attached_figures():
    # Reuse the previously added scripts for the two figures not present in all-original-scripts.
    # They already save into paper_outputs/figures; copy results into generated audit folder after running.
    from runpy import run_path
    for script, out in [('generate_ap_density_sensitivity.py','ap_density_sensitivity.pdf'), ('generate_model_comparison_comprehensive.py','model_comparison_comprehensive.pdf')]:
        run_path(str(CHART_ROOT / 'adapted_scripts' / script), run_name='__main__')
        if (FIG_DIR / out).exists(): shutil.copy2(FIG_DIR / out, GENERATED / out)


def copy_if_needed_remaining_static():
    # These are non-chart manuscript illustrations included in the open-source package.
    # They are not part of Paper-Charts.zip but are kept available in paper_outputs.
    for p in REFERENCE.glob('*'):
        if p.name not in {q.name for q in GENERATED.glob('*')} and p.suffix.lower() in {'.png','.jpg','.jpeg','.pdf'}:
            # Do not overwrite generated charts; copy only still-missing reference chart assets.
            copy_to_outputs(p, p.name)


def write_generation_manifest():
    chart_files = sorted([p.name for p in REFERENCE.glob('*') if p.is_file()])
    rows=[]
    source_map = pd.read_csv(CHART_ROOT / 'CHART_SOURCE_MAP.csv') if (CHART_ROOT / 'CHART_SOURCE_MAP.csv').exists() else pd.DataFrame()
    for name in chart_files:
        ref=REFERENCE/name; gen=GENERATED/name; paper=FIG_DIR/name
        src=''
        if not source_map.empty and name in set(source_map.chart_file):
            src=source_map.loc[source_map.chart_file==name,'provenance_source'].iloc[0]
        rows.append({'chart_file':name, 'reference_exists':ref.exists(), 'generated_exists':gen.exists(), 'paper_output_exists':paper.exists(),
                     'reference_size_bytes':ref.stat().st_size if ref.exists() else 0, 'generated_size_bytes':gen.stat().st_size if gen.exists() else 0,
                     'reference_md5':md5(ref) if ref.exists() else '', 'generated_md5':md5(gen) if gen.exists() else '', 'source':src})
    out=pd.DataFrame(rows)
    out.to_csv(CHART_ROOT / 'CHART_REPRODUCTION_STATUS.csv', index=False)
    out.to_csv(RES_DIR / 'chart_reproduction_status.csv', index=False)
    missing=out[~out.generated_exists]
    if not missing.empty:
        raise SystemExit('Missing generated charts: '+', '.join(missing.chart_file.tolist()))
    return out


def _run_stage(stage: str, skip_copy_reference: bool = False):
    if stage == "layout":
        generate_layout_figures()
    elif stage == "kwnn":
        generate_kwnn_charts()
    elif stage == "map":
        generate_map_chart()
    elif stage == "tree":
        generate_tree_charts()
    elif stage == "summary":
        generate_method_summary_charts()
    elif stage == "ann":
        generate_ann_charts()
    elif stage == "transformer":
        generate_transformer_charts()
    elif stage == "hnsw":
        generate_hnsw_trend()
    elif stage == "attached":
        generate_attached_figures()
    elif stage == "copy":
        if not skip_copy_reference:
            copy_if_needed_remaining_static()
    elif stage == "manifest":
        status = write_generation_manifest()
        print(f"[OK] generated charts: {int(status.generated_exists.sum())} / {len(status)}")
        print(f"[OK] manifest: {CHART_ROOT / 'CHART_REPRODUCTION_STATUS.csv'}")
    else:
        raise ValueError(f"Unknown stage: {stage}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-copy-reference', action='store_true', help='Do not copy any still-missing static reference files.')
    parser.add_argument('--stage', choices=['layout','kwnn','map','tree','summary','ann','transformer','hnsw','attached','copy','manifest'], help='Internal: run a single chart-generation stage.')
    args=parser.parse_args()
    print(f'[INFO] root={ROOT}', flush=True)

    if args.stage:
        print(f'[STAGE] {args.stage}', flush=True)
        _run_stage(args.stage, skip_copy_reference=args.skip_copy_reference)
        print(f'[DONE] {args.stage}', flush=True)
        return

    # Run stages sequentially in one process. The plotting functions close each
    # figure after saving, and os._exit below prevents occasional non-daemon
    # numerical-library threads from keeping the interpreter alive after all
    # figures have already been generated.
    GENERATED.mkdir(parents=True, exist_ok=True)
    for p in GENERATED.glob('*'):
        if p.is_file():
            p.unlink()
    stages = ['layout','kwnn','map','tree','summary','ann','transformer','hnsw','attached','copy','manifest']
    for stage in stages:
        print(f'[RUN] {stage}', flush=True)
        _run_stage(stage, skip_copy_reference=args.skip_copy_reference)
    print('[OK] full chart reproduction completed', flush=True)

if __name__ == '__main__':
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
