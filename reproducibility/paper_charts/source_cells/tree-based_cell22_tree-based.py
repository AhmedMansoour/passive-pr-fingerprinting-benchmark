# Extracted provenance cell for tree-based.pdf
# Source notebook: tree-based.ipynb
# Source cell index: 22
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

# Reconstruct ECDF data
ecdf_data = pd.DataFrame({
    "Error (m)": np.concatenate([
        rf_result_df_m["Error_Euclidean_m"].values,
        xgb_result_df["Error_Euclidean_m"].values,
        catboost_result_df["Error_Euclidean_m"].values
    ]),
    "Method": (["RF"] * len(rf_result_df_m)) +
              (["XGB"] * len(xgb_result_df)) +
              (["CAT"] * len(catboost_result_df))
})

# Reconstruct execution time data
time_data = pd.DataFrame({
    "Execution Time (ms)": np.concatenate([
        rf_result_df_m["ExecutionTime_ms"].values,
        xgb_result_df["ExecutionTime_ms"].values,
        catboost_result_df["ExecutionTime_ms"].values
    ]),
    "Method": (["RF"] * len(rf_result_df_m)) +
              (["XGB"] * len(xgb_result_df)) +
              (["CAT"] * len(catboost_result_df))
})

# Define color palette
palette = {
    "RF": "#1f77b4",
    "XGB": "#2ca02c",
    "CAT": "#ff7f0e"
}

# Create ECDF plot with two inset box plots
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=600)

# Main ECDF
sns.ecdfplot(data=ecdf_data, x="Error (m)", hue="Method", linewidth=0.8, ax=ax, palette=palette)
ax.set_xlabel("Euclidean Error (m)", fontsize=7)
ax.set_ylabel("ECDF", fontsize=7)
ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=6)
ax.grid(True, linestyle=':', linewidth=0.3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

# Inset 1: Time Boxplot
inset_time = fig.add_axes([0.7, 0.60, 0.25, 0.20])
sns.stripplot(data=time_data, x="Method", y="Execution Time (ms)", jitter=True, size=2.5,
              palette=palette, edgecolor='black', linewidth=0.3, ax=inset_time)

# Bar overlay for mean
methods = list(palette.keys())  # ['RF', 'XGB', 'CAT']
means_time = time_data.groupby("Method")["Execution Time (ms)"].mean().reindex(methods)
for i, method in enumerate(methods):
    rgba = list(mcolors.to_rgba(palette[method]))
    rgba[-1] = 0.3  # Transparent
    inset_time.bar(i, means_time[method], width=0.6, color=rgba, zorder=0)

# for i, method in enumerate(means_time.index):
#     rgba = list(mcolors.to_rgba(palette[method]))
#     rgba[-1] = 0.3
#     inset_time.bar(i, means_time[method], width=0.6, color=rgba, zorder=0)
# import matplotlib.colors as mcolors  # Make sure this is imported

# # Bar overlay for mean with transparent color matching the stripplot palette
# for i, method in enumerate(means_time.index):
#     base_color = palette[method]
#     rgba = list(mcolors.to_rgba(base_color))
#     rgba[-1] = 0.3  # Set alpha to 30% transparency
#     inset_time.bar(i, means_time[method], width=0.6, color=rgba, zorder=0)


inset_time.set_xticklabels([])
inset_time.set_xlabel("")
inset_time.set_ylim(0, 30)
inset_time.set_ylabel("Time (ms)", fontsize=6)
inset_time.tick_params(direction='in', length=2, width=0.5, labelsize=6)
inset_time.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.5)
inset_time.set_title("Inference Time", fontsize=6, pad=2)

# Inset 2: Customized Error Boxplot
inset_error = fig.add_axes([0.7, 0.38, 0.25, 0.20])
methods = ["RF", "XGB", "CAT"]
data_to_plot = [ecdf_data[ecdf_data["Method"] == method]["Error (m)"] for method in methods]

bp = inset_error.boxplot(
    data_to_plot,
    patch_artist=True,
    widths=0.5,
    showfliers=False,
    medianprops=dict(color=(0, 0, 0, 0))
)

colors = list(palette.values())
for i, (box, whiskers, caps) in enumerate(zip(
    bp['boxes'],
    zip(bp['whiskers'][::2], bp['whiskers'][1::2]),
    zip(bp['caps'][::2], bp['caps'][1::2])
)):
    rgba = list(mcolors.to_rgba(colors[i]))
    rgba[-1] = 0.3
    box.set_facecolor(rgba)
    box.set_edgecolor(colors[i])
    box.set_linewidth(1.0)

    median_value = np.median(data_to_plot[i])
    inset_error.plot([i + 0.75, i + 1.25], [median_value, median_value],
                     color=colors[i], linestyle='-', linewidth=1.0)

    for whisker in whiskers:
        whisker.set_color(colors[i])
        whisker.set_linestyle('-')
        whisker.set_linewidth(1.0)
    for cap in caps:
        cap.set_color(colors[i])
        cap.set_linewidth(1.0)

inset_error.set_xticks(range(1, len(methods) + 1))
inset_error.set_xticklabels(methods, fontsize=6)
inset_error.set_ylabel("Error (m)", fontsize=6)
inset_error.tick_params(direction='in', length=2, width=0.5, labelsize=6)
inset_error.yaxis.set_major_locator(plt.MultipleLocator(2.0))
inset_error.set_facecolor("white")
inset_error.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.2)

plt.tight_layout()
pdf_path = "tree-based.pdf"
fig.savefig(pdf_path, dpi=1000, bbox_inches='tight', pad_inches=0.01, transparent=False)

plt.show()
