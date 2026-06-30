# Extracted provenance cell for ann_architectures_scaled.pdf
# Source notebook: nn_senstivity.ipynb
# Source cell index: 7
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import numpy as np
from io import StringIO

# ─────────────────────────────────────────────────────────────
# 1) RAW DATA → DF  (removed original 3 and 5; renumbered to 1..4)
# ─────────────────────────────────────────────────────────────
arch_tables = {
    1: """Test_Point  Euclidean_Err_mm  Inference_time_s
b10 707.2922401 0.130268574
b14 2378.532132 0.072618246
b3  794.7953622 0.118522644
b7  1549.554979 0.086501598
d11 2555.807734 0.125595093
d18 3200.843936 0.124389887
d20 5702.006373 0.125677824
d22 3730.900722 0.08682394
d7  4010.006801 0.108930588""",
    2: """Test_Point  Euclidean_Err_mm  Inference_time_s
b10 645.3195235 0.121712923
b14 1939.545512 0.112120152
b3  388.3429784 0.1267941
b7  1969.56109  0.123228788
d11 1733.833543 0.080700874
d18 3592.139961 0.119172335
d20 6491.986502 0.128829002
d22 3623.341025 0.121718407
d7  3683.357405 0.125100136""",
    4: """Test_Point  Euclidean_Err_mm  Inference_time_s
b10 1154.940131 0.114210367
b14 2549.38502  0.10888648
b3  266.4716933 0.073560238
b7  2937.978511 0.119007826
d11 1402.65662  0.118609905
d18 3067.842809 0.111909389
d20 6020.948628 0.122467279
d22 3712.911649 0.124726057
d7  2712.41399  0.119324446""",
    6: """Test_Point  Euclidean_Err_mm  Inference_time_s
b10 811.45023   0.124627113
b14 1947.838383 0.108889341
b3  245.0149106 0.1123631
b7  1936.761334 0.075441599
d11 1501.08701  0.111994505
d18 3581.823738 0.073998451
d20 6624.868713 0.126906157
d22 3532.346721 0.122171879
d7  3392.912077 0.124481916"""
}

# Build dataframe
records = []
for arch, txt in arch_tables.items():
    df_tmp = pd.read_csv(StringIO(txt), sep=r"\s+")
    df_tmp["Architecture"] = arch
    records.append(df_tmp)
df = pd.concat(records, ignore_index=True)

# Continuous renaming {1,2,4,6} → {1,2,3,4}
rename_map = {orig: i+1 for i, orig in enumerate(sorted(arch_tables.keys()))}
df["ArchRenamed"] = df["Architecture"].map(rename_map)

# Units
df["Error_Euclidean"]  = df["Euclidean_Err_mm"] / 1000.0  # m
df["InferenceTime_ms"] = df["Inference_time_s"] * 1000.0  # ms

# ─────────────────────────────────────────────────────────────
# 2) PLOTS  (style aligned to your example)
# ─────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(3.4, 2.0), dpi=500)

# ECDF lines
line_styles = [':', '--', '-.', '-']
line_width  = 1.0
colors      = cm.tab10.colors
arch_vals   = sorted(df["ArchRenamed"].unique())  # [1,2,3,4]

for i, a in enumerate(arch_vals):
    subset = df[df["ArchRenamed"] == a]
    sorted_errors = np.sort(subset["Error_Euclidean"])
    ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax1.plot(sorted_errors, ecdf,
             label=f"Arch {a}",
             linestyle=line_styles[i % len(line_styles)],
             linewidth=line_width,
             color=colors[i % len(colors)])

ax1.set_xlabel("Euclidean Error (m)", fontsize=5)
ax1.set_ylabel("ECDF", fontsize=5)
ax1.tick_params(direction='in', length=3, width=0.6, labelsize=5)
ax1.grid(True, which='both', linestyle='--', linewidth=0.2, alpha=0.4)
ax1.legend(fontsize=6, loc='upper left')

# --- Inset Timing Plot (seaborn strip + mean bars + dashed mean line)
min_time = df["InferenceTime_ms"].min()
max_time = df["InferenceTime_ms"].max()
means = df.groupby("ArchRenamed")["InferenceTime_ms"].mean().sort_index()
x_positions = range(len(means))

inset_timing = fig.add_axes([0.60, 0.56, 0.35, 0.23])
sns.stripplot(
    data=df, x="ArchRenamed", y="InferenceTime_ms",
    jitter=True, size=2.5, palette="tab10",
    edgecolor='black', linewidth=0.3, ax=inset_timing
)

# mean bars and dashed trend
for i, (xpos, mean_val) in enumerate(zip(x_positions, means)):
    rgba = list(colors[i % len(colors)]) + [0.30]
    inset_timing.bar(xpos, mean_val, width=0.6, color=rgba, edgecolor=None, zorder=0)
inset_timing.plot(x_positions, means.values, linewidth=0.7, linestyle='--')

inset_timing.set_xticklabels([])  # no x tick labels in inset
inset_timing.set_xlabel("")
inset_timing.set_ylim(bottom=min_time - 0.01 * min_time,
                      top=max_time + 0.01 * max_time)
inset_timing.set_ylabel("Time (ms)", fontsize=5)
inset_timing.tick_params(direction='in', length=2, width=0.5, labelsize=5)
inset_timing.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
inset_timing.set_title("Inference Time (ms)", fontsize=5, pad=2)

# --- Inset Boxplot (no outliers, colored boxes, manual medians)
inset_ax = fig.add_axes([0.60, 0.27, 0.35, 0.23])
data_to_plot = [df[df['ArchRenamed'] == a]["Error_Euclidean"] for a in arch_vals]
tick_labels  = [f"Arch {a}" for a in arch_vals]

bp = inset_ax.boxplot(
    data_to_plot,
    patch_artist=True,
    widths=0.5,
    showfliers=False,
    medianprops=dict(color=(0, 0, 0, 0))
)

for i, (box, whiskers, caps) in enumerate(zip(
    bp['boxes'],
    zip(bp['whiskers'][::2], bp['whiskers'][1::2]),
    zip(bp['caps'][::2], bp['caps'][1::2])
)):
    color = colors[i % len(colors)]
    rgba  = list(color) + [0.30]
    box.set_facecolor(rgba)
    box.set_edgecolor(color)
    box.set_linewidth(1.0)

    # manual median line
    med_val = np.median(data_to_plot[i])
    inset_ax.plot([i + 0.75, i + 1.25], [med_val, med_val],
                  color=color, linestyle='-', linewidth=1.0)

    for w in whiskers:
        w.set_color(color)
        w.set_linestyle('-')
        w.set_linewidth(1.0)
    for c in caps:
        c.set_color(color)
        c.set_linewidth(1.0)

inset_ax.set_xticks(range(1, len(tick_labels) + 1))
inset_ax.set_xticklabels(tick_labels, fontsize=5)
inset_ax.set_ylabel("Error (m)", fontsize=5)
inset_ax.set_title("Box Plots (Without Outliers)", fontsize=5, pad=2)
inset_ax.tick_params(direction='in', length=2, width=0.5, labelsize=5)
inset_ax.yaxis.set_major_locator(MultipleLocator(1.0))
inset_ax.set_facecolor("white")
inset_ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.2)

# Main ECDF axes limits (tune as needed)
ax1.set_xlim(0, 8)
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.set_ylim(0, 1.02)
ax1.yaxis.set_major_locator(MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(MultipleLocator(0.2))

plt.tight_layout()
fig.savefig("ann_architectures_scaled.pdf",
            dpi=1000, bbox_inches='tight', pad_inches=0.01, transparent=False)
plt.show()
