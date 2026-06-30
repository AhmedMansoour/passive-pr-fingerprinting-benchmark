# Extracted provenance cell for methods_bxp_statistical.pdf
# Source notebook: df_all_methods.ipynb
# Source cell index: 2
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# Build the figure directly from in-memory `df_all_methods`
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.linewidth": 0.3
})

# -------------------------------
# 0) Sanity & numeric coercion
# -------------------------------
if "df_all_methods" not in globals():
    raise RuntimeError("`df_all_methods` not found in memory.")

num_cols = ["Mean","Std","Min","Q1","Median","Q3","Max","Time_ms","TimeStd_ms"]
dfp = df_all_methods.copy()
for c in num_cols:
    if c in dfp.columns:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

# -----------------------------------------
# 1) Choose methods in desired plot order
#    (KWNN k=6 Avg/KF, then MAP k=6,
#     then Trees, ANN, Transformer)
# -----------------------------------------
chosen_methods = [
    "KWNN k=6 (Avg)",
    "KWNN k=6 (KF)",
    "MAP k=6",
    "Random Forest (RF)",
    "XGBoost (XGB)",
    "CatBoost (CAT)",
    "ANN Arch 1",
    "ANN Arch 2",
    "ANN Arch 3",
    "ANN Arch 4",
    "TF Arch 1",
    "TF Arch 2",
    "TF Arch 3",
    "TF Arch 4",
]

df_sub = dfp[dfp["Method"].isin(chosen_methods)].copy()
cat_type = pd.CategoricalDtype(categories=chosen_methods, ordered=True)
df_sub["Method"] = df_sub["Method"].astype(cat_type)
df_sub = df_sub.sort_values("Method").reset_index(drop=True)

# ---------------------------------------------------------
# 2) Merge families: KWNN + MAP -> "Statistical" family
# ---------------------------------------------------------
df_sub["Family"] = df_sub["Family"].replace({
    "KWNN": "Statistical",
    "MAP": "Statistical",
    "KWNN Auto-k": "Statistical",  # harmless if not present
})

# ---------------------------------------------------------
# 3) Prepare stats for matplotlib.bxp
# ---------------------------------------------------------
stats, families = [], []
for _, row in df_sub.iterrows():
    med = row["Median"]; q1 = row["Q1"]; q3 = row["Q3"]
    lo  = row["Min"];    hi = row["Max"]
    # Fallbacks if some quartiles are missing
    q1 = q1 if pd.notna(q1) else med
    q3 = q3 if pd.notna(q3) else med
    lo = lo if pd.notna(lo) else med
    hi = hi if pd.notna(hi) else med

    stats.append({
        "label": row["Method"],
        "med": float(med),
        "q1":  float(q1),
        "q3":  float(q3),
        "whislo": float(lo),
        "whishi": float(hi),
    })
    families.append(row["Family"])

# ---------------------------------------------------------
# 4) Family colors (shared within family)
# ---------------------------------------------------------
family_colors = {
    "Statistical": (0.16, 0.37, 0.66),  # deep blue for KWNN + MAP
    "Trees":       (0.20, 0.57, 0.20),  # green
    "ANN":         (0.80, 0.49, 0.13),  # orange
    "Transformer": (0.45, 0.16, 0.60),  # purple
}
families_in_order = families  # aligned with stats

# ---------------------------------------------------------
# 5) Label wrapper (rotation=0, multi-line)
# ---------------------------------------------------------
def wrap_label(lbl, width=12):
    import re, textwrap
    s = str(lbl).strip()

    # Put KWNN on two lines: "KWNN\nk=6 (Avg)" etc.
    if s.startswith("KWNN k=") and "(" in s:
        m = re.match(r"KWNN\s+(k=\d+)\s+\(([^)]+)\)", s)
        if m:
            return f"KWNN\n{m.group(1)} ({m.group(2)})"

    # MAP -> "MAP\nk=6"
    if s.startswith("MAP k="):
        return s.replace(" ", "\n", 1)

    # Trees short names on two lines
    if s.lower().startswith("random forest"):
        return "Random\nForest"
    if s.startswith("XGBoost"):
        return "XGBoost\n(XGB)"
    if s.startswith("CatBoost"):
        return "CatBoost\n(CAT)"

    # ANN/TF: break before "Arch"
    if " Arch " in s:
        return s.replace(" Arch ", "\nArch ")

    # Generic wrapper fallback
    w = textwrap.wrap(s, width=width)
    return "\n".join(w[:2]) if w else s

labels_wrapped = [wrap_label(s["label"]) for s in stats]

# ---------------------------------------------------------
# 6) Plot
# ---------------------------------------------------------
os.makedirs("/mnt/data/figures", exist_ok=True)

n = len(stats)
x = np.arange(n, dtype=float)
box_width = 0.6
fig_w = max(6.0, 0.52 * n)
fig_h = 2.8

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=600)
bp = ax.bxp(stats, positions=x, widths=box_width, showfliers=False, patch_artist=True)

# Style boxes/whiskers/caps with family colors
for i, box in enumerate(bp["boxes"]):
    fam = families_in_order[i]
    base = family_colors.get(fam, (0.5, 0.5, 0.5))
    box.set_facecolor((*base, 0.28))
    box.set_edgecolor(base)
    box.set_linewidth(1.0)
for i in range(len(bp["whiskers"])//2):
    fam = families_in_order[i]
    base = family_colors.get(fam, (0.4, 0.4, 0.4))
    for w in bp["whiskers"][2*i:2*i+2]:
        w.set_color(base); w.set_linewidth(1.0)
    for c in bp["caps"][2*i:2*i+2]:
        c.set_color(base); c.set_linewidth(1.0)

# Hide default medians; draw short family-colored tick instead
for med in bp["medians"]:
    med.set_alpha(0)

meds = [s["med"] for s in stats]
mxs  = [s["whishi"] for s in stats]
for i, med in enumerate(meds):
    fam = families_in_order[i]
    base = family_colors.get(fam, (0.4, 0.4, 0.4))
    ax.plot([x[i] - box_width/4, x[i] + box_width/4], [med, med], color=base, linewidth=1.2, zorder=2.2)

# Dashed connectors
ax.plot(x, meds, linestyle="--", marker="o", markersize=2.2, linewidth=0.9, color="blue",  label="Medians")
ax.plot(x, mxs,  linestyle="--", marker="o", markersize=2.2, linewidth=0.9, color="black", label="Maxima")

# Axes cosmetics
ax.set_xticks(x)
ax.set_xticklabels(labels_wrapped, rotation=0, ha="center", fontsize=7)
plt.subplots_adjust(bottom=0.30)
ax.set_ylabel("Euclidean error (m)", fontsize=8)
ax.tick_params(which="both", direction="in", top=True, right=True, length=3, width=0.6, labelsize=7)
ax.yaxis.set_minor_locator(MultipleLocator(1.0))
ax.yaxis.set_major_locator(MultipleLocator(2.0))
ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.3, alpha=0.35)

# Legends
from matplotlib.patches import Patch
handles = [Patch(facecolor=(*clr,0.28), edgecolor=clr, label=fam)
           for fam, clr in family_colors.items() if fam in df_sub["Family"].unique()]
leg1 = ax.legend(handles=handles, fontsize=7, loc="upper center", ncol=5, frameon=True)
ax.add_artist(leg1)
ax.legend(fontsize=7, loc="lower right", frameon=True)

plt.tight_layout()
fig_out = "methods_bxp_statistical.pdf"
fig.savefig(fig_out, dpi=1000, bbox_inches="tight", pad_inches=0.01)
fig_out
