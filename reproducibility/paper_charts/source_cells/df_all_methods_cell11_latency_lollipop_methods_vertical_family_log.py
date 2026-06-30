# Extracted provenance cell for latency_lollipop_methods_vertical_family_log.pdf
# Source notebook: df_all_methods.ipynb
# Source cell index: 11
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# Lollipop latency chart (methods on x-axis) incl. HNSW
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

# --- prep ---
dfp = df_all_methods.copy()
dfp["Time_ms"] = pd.to_numeric(dfp["Time_ms"], errors="coerce")

chosen_methods = [
    # Statistical (KWNN)
    "KWNN k=6 (Avg)",
    "KWNN k=6 (KF)",
    "KWNN k=6 (HNSW)",  # ← added
    # Trees
    "Random Forest (RF)", "XGBoost (XGB)", "CatBoost (CAT)",
    # ANN
    "ANN Arch 1","ANN Arch 2","ANN Arch 3","ANN Arch 4",
    # Transformer
    "TF Arch 1","TF Arch 2","TF Arch 3","TF Arch 4",
]
df_sub = dfp[dfp["Method"].isin(chosen_methods)].copy()
df_sub = df_sub[df_sub["Time_ms"].notna()].copy()

# Map KWNN to "Statistical" (legend/color)
df_sub["FamilyMapped"] = df_sub["Family"].replace({"KWNN": "Statistical"})

# Sort by time (slowest→fastest so HNSW shows at the right end)
df_sorted = df_sub.sort_values("Time_ms", ascending=False).reset_index(drop=True)

# --- wrapped x labels (rotation 0) ---
def wrap_label(lbl):
    s = str(lbl).strip()
    if s.lower().startswith("random forest"):
        return "Random\nForest"
    if s.startswith("KWNN k=") and "(" in s:
        m = re.match(r"KWNN\s+(k=\d+)\s+\(([^)]+)\)", s)
        if m: return f"KWNN\n{m.group(1)} ({m.group(2)})"
    if " (" in s and s.endswith(")"):
        return s.replace(" (", "\n(")
    if " Arch " in s:
        return s.replace(" Arch ", "\nArch ")
    return s

df_sorted["Wrapped"] = df_sorted["Method"].apply(wrap_label)

# --- family colors ---
family_colors = {
    "Statistical": (0.16, 0.37, 0.66),  # deep blue
    "Trees":       (0.20, 0.57, 0.20),  # green
    "ANN":         (0.80, 0.49, 0.13),  # orange
    "Transformer": (0.45, 0.16, 0.60),  # purple
}
colors = [family_colors.get(f, (0.5,0.5,0.5)) for f in df_sorted["FamilyMapped"]]

# --- plot (linear y) ---
os.makedirs("figures", exist_ok=True)
x = np.arange(len(df_sorted))
times = df_sorted["Time_ms"].to_numpy(float)

plt.rcParams.update({"axes.linewidth": 0.3})
# fig, ax = plt.subplots(figsize=(max(6.4, 0.58*len(df_sorted)), 2.2), dpi=300)

# # stems + markers (family-colored stems, black markers)
# for i, t in enumerate(times):
#     ax.plot([x[i], x[i]], [0, t], linewidth=1.0, color=colors[i])
# ax.plot(x, times, linestyle="", marker="o", markersize=3.2, color="black", label="Mean time (ms)")

# # cosmetics
# ax.set_xticks(x)
# ax.set_xticklabels(df_sorted["Wrapped"], rotation=0, ha="center", fontsize=7)
# plt.subplots_adjust(bottom=0.30)
# ax.set_ylabel("Mean inference time (ms)", fontsize=8)
# ax.tick_params(which="both", direction="in", top=True, right=True, length=3, width=0.6, labelsize=7)
# ax.yaxis.set_minor_locator(MultipleLocator(5.0))
# ax.yaxis.set_major_locator(MultipleLocator(20.0))
# ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.3, alpha=0.35)

# # family legend
# handles = []
# for fam in ["Statistical","Trees","ANN","Transformer"]:
#     if fam in set(df_sorted["FamilyMapped"]):
#         clr = family_colors[fam]
#         handles.append(Patch(facecolor=(*clr,0.28), edgecolor=clr, label=fam))
# ax.legend(handles=handles, title="Family", fontsize=7, loc="upper right", frameon=True).get_frame().set_linewidth(0.3)

# plt.tight_layout()
# out_path = "figures/latency_lollipop_methods_vertical_family.png"
# plt.savefig(out_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
# print("Saved:", out_path)

# --- optional: log-scale version (safe for 0 ms by clamping to epsilon) ---
eps = 1e-3  # ms
times_log = times.copy()
times_log[times_log <= 0] = eps

fig2, ax2 = plt.subplots(figsize=(max(6.4, 0.58*len(df_sorted)), 2.4), dpi=300)
for i, t in enumerate(times_log):
    ax2.plot([x[i], x[i]], [eps, t], linewidth=1.0, color=colors[i])
ax2.plot(x, times_log, linestyle="", marker="o", markersize=3.2, color="black")
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(df_sorted["Wrapped"], rotation=0, ha="center", fontsize=7)
plt.subplots_adjust(bottom=0.30)
ax2.set_ylabel("Mean inference time (ms, log)", fontsize=8)
ax2.tick_params(which="both", direction="in", top=True, right=True, length=3, width=0.6, labelsize=7)
ax2.grid(True, axis="y", which="both", linestyle="--", linewidth=0.3, alpha=0.35)
ax2.legend(handles=handles, ncol=4, fontsize=7, loc="upper right", frameon=True).get_frame().set_linewidth(0.3)

# Optional: annotate clamped zeros
if np.any(df_sorted["Time_ms"] <= 0):
    ix = np.where(times <= 0)[0]
    for j in ix:
        ax2.annotate("≈0 ms", (x[j], eps), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontsize=6)

plt.tight_layout()
out_path_log = "latency_lollipop_methods_vertical_family_log.pdf"
plt.savefig(out_path_log, bbox_inches="tight", pad_inches=0.01, dpi=500)
print("Saved:", out_path_log)
