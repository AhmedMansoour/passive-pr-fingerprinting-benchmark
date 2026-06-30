# Extracted provenance cell for lollipop_median_desc_familycolors_VERTICAL.pdf
# Source notebook: df_all_methods.ipynb
# Source cell index: 8
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# Family-colored vertical lollipop chart for MEDIAN error (sorted high->low),
# using the in-memory df_all_methods and the same chosen subset of methods.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.linewidth": 0.3
})

# --- Sanity check ---
if "df_all_methods" not in globals():
    raise RuntimeError("`df_all_methods` not found in this session. Please define it and re-run.")

# Ensure numeric
num_cols = ["Mean","Std","Min","Q1","Median","Q3","Max","Time_ms","TimeStd_ms"]
dfp = df_all_methods.copy()
for c in num_cols:
    if c in dfp.columns:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

# Chosen subset (same as earlier figures)
chosen_methods = [
    # Statistical/KWNN (k=6 only)
    "KWNN k=6 (Avg)",
    "KWNN k=6 (KF)",
    # Trees
    "Random Forest (RF)",
    "XGBoost (XGB)",
    "CatBoost (CAT)",
    # ANN
    "ANN Arch 1",
    "ANN Arch 2",
    "ANN Arch 3",
    "ANN Arch 4",
    # Transformer
    "TF Arch 1",
    "TF Arch 2",
    "TF Arch 3",
    "TF Arch 4",
]

df_sub = dfp[dfp["Method"].isin(chosen_methods)].copy()

# Keep order for deterministic mapping, but we'll sort by Median for plotting order
cat_type = pd.CategoricalDtype(categories=chosen_methods, ordered=True)
df_sub["Method"] = df_sub["Method"].astype(cat_type)

# --- Family normalization (KWNN -> Statistical) ---
# Keep compatibility: if either exists, normalize to "Statistical"
df_sub["Family"] = df_sub["Family"].replace({"KWNN": "Statistical"})

# Sort by Median (descending)
df_sorted = df_sub.sort_values("Median", ascending=False).reset_index(drop=True)

# --- Family colors (same across charts) ---
family_colors = {
    "Statistical": (0.16, 0.37, 0.66),  # deep blue
    "Trees":       (0.20, 0.57, 0.20),  # green
    "ANN":         (0.80, 0.49, 0.13),  # orange
    "Transformer": (0.45, 0.16, 0.60),  # purple
}

# --- Label wrapper ---
def wrap_label(lbl, width=12):
    s = str(lbl).strip()
    # Special case: Random Forest label
    low = s.lower()
    if low.startswith("random forest"):
        return "Random\nForest"
    # KWNN -> custom two-line
    if s.startswith("KWNN k=") and "(" in s:
        import re
        m = re.match(r"KWNN\s+(k=\d+)\s+\(([^)]+)\)", s)
        if m:
            return f"KWNN\n{m.group(1)} ({m.group(2)})"
    # Generic patterns
    if " (" in s and s.endswith(")"):
        return s.replace(" (", "\n(")
    if " Arch " in s:
        return s.replace(" Arch ", "\nArch ")
    import textwrap
    w = textwrap.wrap(s, width=width)
    return "\n".join(w[:2]) if w else s

df_sorted["Wrapped"] = df_sorted["Method"].map(wrap_label)

# --- Build plot (vertical lollipop: methods on x-axis, medians as stems) ---
os.makedirs("/mnt/data/figures", exist_ok=True)

xpos = np.arange(len(df_sorted))
meds = df_sorted["Median"].to_numpy(dtype=float)

fig_w = max(6.0, 0.56 * len(df_sorted))
fig_h = 2
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)

# Draw stems+markers per method with its family color
for i, row in df_sorted.iterrows():
    fam = row["Family"]
    base = family_colors.get(fam, (0.4, 0.4, 0.4))
    # stem
    ax.plot([xpos[i], xpos[i]], [0, meds[i]], linewidth=1.0, color=base, zorder=1.0)
    # marker
    ax.plot([xpos[i]], [meds[i]], marker="o", linestyle="", markersize=3.2, color=base, zorder=1.5)

# Cosmetics
ax.set_xticks(xpos)
ax.set_xticklabels(df_sorted["Wrapped"].tolist(), rotation=0, ha="center", fontsize=7)
plt.subplots_adjust(bottom=0.28)  # extra room for wrapped labels
ax.set_ylabel("Median error (m)", fontsize=8)
ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=7)
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.3, alpha=0.35)

# Family legend
from matplotlib.patches import Patch
handles = []
for fam in df_sorted["Family"].unique():
    clr = family_colors.get(fam, (0.4, 0.4, 0.4))
    handles.append(Patch(facecolor=clr, edgecolor=clr, label=fam))
leg = ax.legend(handles=handles, ncol=4, fontsize=7, loc="upper right", frameon=True)
if leg and leg.get_frame():
    leg.get_frame().set_linewidth(0.3)

plt.tight_layout()
out_path = "lollipop_median_desc_familycolors_VERTICAL.pdf"
fig.savefig(out_path, dpi=1000, bbox_inches="tight", pad_inches=0.01)
out_path
