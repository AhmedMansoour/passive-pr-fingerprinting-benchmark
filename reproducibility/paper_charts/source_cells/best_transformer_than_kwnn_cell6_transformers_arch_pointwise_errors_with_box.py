# Extracted provenance cell for transformers_arch_pointwise_errors_with_box.pdf
# Source notebook: best transformer than kwnn.ipynb
# Source cell index: 6
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# ============================================================
# Point-wise errors by architecture with overlaid boxplots
# + blue dashed line for medians
# + red dashed line for maxima
# Requires: all_stage_perpoint (dict of per-point DataFrames)
# Output:  figures/arch_pointwise_errors_with_box.pdf
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

# 1) Gather per-point errors (m) for the four architectures
arch_keys = [
    ("Arch 1", "Arch 1 — TF (scaled)"),
    ("Arch 2", "Arch 2 — TF + CosWR"),
    ("Arch 3", "Arch 3 — AE+TF (λ=0, σ=0)"),
    ("Arch 4", "Arch 4 — AE+TF (λ=0.5, σ=0.1)"),
]

frames = []
for short, key in arch_keys:
    if key not in all_stage_perpoint:
        continue
    df = all_stage_perpoint[key].copy()
    # ensure error in meters
    if "error_m" not in df and "error_mm" in df:
        df["error_m"] = df["error_mm"] / 1000.0
    df = df.rename(columns={"Point": "TestPoint"})
    frames.append(df[["TestPoint", "error_m"]].assign(Architecture=short))

if not frames:
    raise RuntimeError("No per-point DataFrames found in all_stage_perpoint for Arch 1..4.")

df_long = pd.concat(frames, ignore_index=True)

# 2) Keep only test points common to all present architectures
present_arches = sorted(df_long["Architecture"].unique(), key=lambda a: int(a.split()[1]))  # Arch 1..4
counts = df_long.groupby("TestPoint")["Architecture"].nunique()
common_points = counts[counts == len(present_arches)].index
df_long = df_long[df_long["TestPoint"].isin(common_points)].copy()

# 3) Plot setup
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(3.8, 2.0), dpi=600)

# Categorical x positions per architecture
xpos = {a: i for i, a in enumerate(present_arches)}
x_vals = np.array([xpos[a] for a in present_arches])

# Colors per test point (repeat tab10)
point_ids = sorted(df_long["TestPoint"].unique())
colors = cm.tab10.colors
color_map = {tp: colors[i % len(colors)] for i, tp in enumerate(point_ids)}

# 4) Overlaid boxplots (behind scatter)
data_by_arch = [df_long.loc[df_long["Architecture"] == a, "error_m"].to_numpy()
                for a in present_arches]

box_width = 0.32
bp = ax.boxplot(
    data_by_arch,
    positions=x_vals,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color=(0, 0, 0, 0))  # hide default median
)

arch_palette = [cm.tab10(i) for i in range(len(present_arches))]

# style each box, whiskers, caps
for i, box in enumerate(bp['boxes']):
    base = arch_palette[i]
    rgba = (base[0], base[1], base[2], 0.30)  # translucent fill
    box.set_facecolor(rgba)
    box.set_edgecolor(base)
    box.set_linewidth(1.0)
    box.set_zorder(1.5)

for i in range(len(present_arches)):
    for w in bp['whiskers'][2*i:2*i+2]:
        w.set_color(arch_palette[i]); w.set_linewidth(1.0); w.set_linestyle('-'); w.set_zorder(1.5)
    for c in bp['caps'][2*i:2*i+2]:
        c.set_color(arch_palette[i]); c.set_linewidth(1.0); c.set_zorder(1.5)

# 5) Compute medians & maxima, plot lines
medians, maxima = [], []
for i, a in enumerate(present_arches):
    med = np.median(data_by_arch[i]) if len(data_by_arch[i]) else np.nan
    mx  = np.max(data_by_arch[i]) if len(data_by_arch[i]) else np.nan
    medians.append(med)
    maxima.append(mx)
    # manual median line
    ax.plot([x_vals[i] - box_width/4, x_vals[i] + box_width/4],
            [med, med], color=arch_palette[i], linewidth=1.2, zorder=2.2)

# blue dashed line = medians
ax.plot(x_vals, medians, linestyle='--', marker='o', markersize=1.5,
        color='blue', linewidth=0.5, zorder=2.6, label="Medians")

# red dashed line = maxima
ax.plot(x_vals, maxima, linestyle='--', marker='o', markersize=1.5,
        color='k', linewidth=0.5, zorder=2.6, label="Maxima")

# 6) Jittered scatter points (on top)
rng = np.random.default_rng(0)
jitter_width = 0.18
marker_size = 10

for a in present_arches:
    sub = df_long[df_long["Architecture"] == a]
    x_base = xpos[a]
    x_jit = x_base + (rng.random(len(sub)) - 0.5) * 2 * jitter_width
    y = sub["error_m"].values
    c = [color_map[tp] for tp in sub["TestPoint"].values]
    # ax.scatter(x_jit, y, s=marker_size, c=c, edgecolor="black",
    #            linewidths=0.25, zorder=3)

# 7) Cosmetics
ax.set_xticks(x_vals)
ax.set_xticklabels(present_arches, fontsize=6)
ax.set_ylabel("Euclidean Error (m)", fontsize=6)

# Optional y limits
# ax.set_ylim(0, 8)

# Minor ticks inside; ticks on all sides
ax.tick_params(which="both", direction="in", top=True, right=True,
               length=3, width=0.6, labelsize=6)
ax.yaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(2.0))
ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.3, alpha=0.35)

# Add legend for med/max lines
leg = ax.legend(fontsize=5, loc="best", frameon=True)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(0.3)

plt.tight_layout()
out_path = "transformers_arch_pointwise_errors_with_box.pdf"
fig.savefig(out_path, dpi=1000, bbox_inches="tight", pad_inches=0.01, transparent=False)
plt.show()
print(f"Saved: {out_path}")
