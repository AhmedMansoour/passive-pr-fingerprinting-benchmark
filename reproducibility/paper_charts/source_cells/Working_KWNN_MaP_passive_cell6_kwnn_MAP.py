# Extracted provenance cell for kwnn_MAP.pdf
# Source notebook: Working_KWNN_MaP_passive.ipynb
# Source cell index: 6
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import numpy as np



fig, ax1 = plt.subplots(figsize=(3.4, 2.4), dpi=500)

# --- ECDF plot by k ---
line_styles = [':', '--', '-.', '-']
line_width = 1.0
colors = cm.tab10.colors
k_values = sorted(sensitivity_results['k'].unique())

for i, k in enumerate(k_values):
    subset = sensitivity_results[sensitivity_results['k'] == k]
    sorted_errors = np.sort(subset["Error_Euclidean"])
    ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax1.plot(sorted_errors, ecdf,
             label=f"k={k}",
             linestyle=line_styles[i % len(line_styles)],
             linewidth=line_width,
             color=colors[i % len(colors)])

ax1.set_xlabel("Euclidean Error (m)", fontsize=5)
ax1.set_ylabel("ECDF", fontsize=5)
ax1.tick_params(direction='in', length=3, width=0.6, labelsize=5)
ax1.grid(True, which='both', linestyle='--', linewidth=0.2, alpha=0.4)
ax1.legend(fontsize=7, loc='upper left')

# --- Inset Timing Plot ---
min_time = sensitivity_results["InferenceTime_ms"].min()
max_time = sensitivity_results["InferenceTime_ms"].max()
means = sensitivity_results.groupby("k")["InferenceTime_ms"].mean().sort_index()
x_positions = range(len(means))

inset_timing = fig.add_axes([0.6, 0.56, 0.35, 0.23])
sns.stripplot(data=sensitivity_results, x="k", y="InferenceTime_ms",
              jitter=True, size=2.5, palette="tab10", edgecolor='black', linewidth=0.3, ax=inset_timing)

for i, (xpos, mean_val) in enumerate(zip(x_positions, means)):
    rgba = list(colors[i % len(colors)]) + [0.3]
    inset_timing.bar(xpos, mean_val, width=0.6, color=rgba, edgecolor=None, zorder=0)

inset_timing.plot(x_positions, means.values, color='blue', linewidth=.7, linestyle='--')
inset_timing.set_xticklabels([])
inset_timing.set_xlabel("")
inset_timing.set_ylim(bottom=min_time - 0.01 * min_time, top=max_time + 0.01 * max_time)
inset_timing.set_ylabel("Time (ms)", fontsize=5)
inset_timing.tick_params(direction='in', length=2, width=0.5, labelsize=5)
inset_timing.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
inset_timing.set_title("Inference Time (ms)", fontsize=5, pad=2)

# --- Inset Boxplot ---
inset_ax = fig.add_axes([0.6, 0.27, 0.35, 0.23])
data_to_plot = [sensitivity_results[sensitivity_results['k'] == k]["Error_Euclidean"] for k in k_values]
tick_labels = [f"k={k}" for k in k_values]

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
    rgba = list(color) + [0.3]
    box.set_facecolor(rgba)
    box.set_edgecolor(color)
    box.set_linewidth(1.0)

    median_value = np.median(data_to_plot[i])
    inset_ax.plot([i + 0.75, i + 1.25], [median_value, median_value],
                  color=color, linestyle='-', linewidth=1.0)

    for whisker in whiskers:
        whisker.set_color(color)
        whisker.set_linestyle('-')
        whisker.set_linewidth(1.0)
    for cap in caps:
        cap.set_color(color)
        cap.set_linewidth(1.0)

inset_ax.set_xticks(range(1, len(tick_labels) + 1))
inset_ax.set_xticklabels(tick_labels, fontsize=5)
inset_ax.set_ylabel("Error (m)", fontsize=5)
inset_ax.set_title("Box Plots (Without Outliers)", fontsize=5, pad=2)
inset_ax.tick_params(direction='in', length=2, width=0.5, labelsize=5)
inset_ax.yaxis.set_major_locator(MultipleLocator(1.0))
inset_ax.set_facecolor("white")
inset_ax.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.2)

plt.tight_layout()
pdf_path = "kwnn_MAP.pdf"
fig.savefig(pdf_path, dpi=1000, bbox_inches='tight', pad_inches=0.01, transparent=False)

plt.show()
