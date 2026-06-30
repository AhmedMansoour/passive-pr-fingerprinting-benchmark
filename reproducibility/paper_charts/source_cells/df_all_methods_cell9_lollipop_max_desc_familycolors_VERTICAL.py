# Extracted provenance cell for lollipop_max_desc_familycolors_VERTICAL.pdf
# Source notebook: df_all_methods.ipynb
# Source cell index: 9
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# Max-error lollipop (vertical, methods on x-axis) with family colors, built from `df_all_methods`.
# - Uses the SAME family colors as the median plot
# - Sorted high -> low by Max
# - Multi-line x tick labels (no rotation) to avoid overlap

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.linewidth": 0.3
})

# -------------------------
# 0) Sanity & prep
# -------------------------
if "df_all_methods" not in globals():
    raise RuntimeError("`df_all_methods` not found in this session. Please define it before running.")

dfp = df_all_methods.copy()
for c in ["Mean","Std","Min","Q1","Median","Q3","Max","Time_ms","TimeStd_ms"]:
    if c in dfp.columns:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

# Map KWNN + MAP families under "Statistical" for color grouping
if "Family" in dfp.columns:
    dfp["Family"] = dfp["Family"].replace({"KWNN": "Statistical", "MAP": "Statistical"})

# Choose the same set of methods used in the (previous) median plot.
# If some are missing, we'll silently drop.
wanted = [
    "KWNN k=6 (Avg)",
    "KWNN k=6 (KF)",
    "Random Forest (RF)",
    "XGBoost (XGB)",
    "CatBoost (CAT)",
    "ANN Arch 1", "ANN Arch 2", "ANN Arch 3", "ANN Arch 4",
    "TF Arch 1",  "TF Arch 2",  "TF Arch 3",  "TF Arch 4",
    # include MAP top-k k=6 if present
    "MAP top-k (k=6)"
]
exists = [m for m in wanted if m in dfp["Method"].unique()]

df_plot = dfp[dfp["Method"].isin(exists)].copy()

# Sort by Max descending
df_plot = df_plot.sort_values("Max", ascending=False).reset_index(drop=True)

# -------------------------
# Wrap labels for x-axis
# -------------------------
import re, textwrap
def wrap_label(lbl, width=12):
    s = str(lbl).strip()
    # Random Forest -> "Random\nForest"
    if s.lower().startswith("random forest"):
        return "Random\nForest"
    # KWNN k=<n> (...)
    m = re.match(r"KWNN\s+(k=\d+)\s+\(([^)]+)\)", s)
    if m:
        return f"KWNN\n{m.group(1)} ({m.group(2)})"
    # Generic "(...)" -> linebreak before "("
    if " (" in s and s.endswith(")"):
        return s.replace(" (", "\n(")
    # " Arch " -> linebreak
    if " Arch " in s:
        return s.replace(" Arch ", "\nArch ")
    w = textwrap.wrap(s, width=width)
    return "\n".join(w[:2]) if w else s

df_plot["Wrapped"] = df_plot["Method"].map(wrap_label)

# -------------------------
# Family colors
# -------------------------
family_colors = {
    "Statistical": (0.16, 0.37, 0.66),  # deep blue for KWNN+MAP
    "Trees":       (0.20, 0.57, 0.20),  # green
    "ANN":         (0.80, 0.49, 0.13),  # orange
    "Transformer": (0.45, 0.16, 0.60),  # purple
}
# fallback for any legacy family names
fallback = (0.4, 0.4, 0.4)

# -------------------------
# Plot (vertical lollipop)
# -------------------------
os.makedirs("/mnt/data/figures", exist_ok=True)

n = len(df_plot)
x = np.arange(n, dtype=float)
mxs = df_plot["Max"].to_numpy(dtype=float)

fig_w = max(6.0, 0.55 * n)
fig_h = 2.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)

# stems + markers with family colors
for i in range(n):
    fam = df_plot.loc[i, "Family"]
    base = family_colors.get(fam, fallback)
    ax.plot([x[i], x[i]], [0, mxs[i]], linewidth=1.0, color=base, alpha=0.9, zorder=1)
    ax.plot([x[i]], [mxs[i]], marker="o", linestyle="", markersize=3.2, color=base, zorder=2)

# Cosmetics
ax.set_xticks(x)
ax.set_xticklabels(df_plot["Wrapped"].tolist(), rotation=0, ha="center", fontsize=7)
ax.set_ylabel("Max error (m)", fontsize=8)
ax.tick_params(which="both", direction="in", top=True, right=True, length=3, width=0.6, labelsize=7)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.3, alpha=0.35)

# Family legend
from matplotlib.patches import Patch
present_fams = df_plot["Family"].dropna().unique().tolist()
handles = [Patch(facecolor=(*family_colors.get(f, fallback), 0.6), edgecolor=family_colors.get(f, fallback), label=f) for f in present_fams]
leg = ax.legend(handles=handles, ncol=4, fontsize=7, loc="upper right", frameon=True)
leg.get_frame().set_linewidth(0.3)

plt.tight_layout()
out_path = "lollipop_max_desc_familycolors_VERTICAL.pdf"
fig.savefig(out_path, dpi=1000, bbox_inches="tight", pad_inches=0.01)
out_path
