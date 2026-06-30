# Extracted provenance cell for knn-based_xy.pdf
# Source notebook: KWNN XY LOCATIONS.ipynb
# Source cell index: 4
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt
import numpy as np

# Prepare the data for plotting from timed_comparison_df
# We'll plot one color per k value

# Rename columns for clarity
timed_comparison_df["X_true_m"] = timed_comparison_df["X_true"]
timed_comparison_df["Y_true_m"] = timed_comparison_df["Y_true"]
timed_comparison_df["X_pred_m_KNN"] = timed_comparison_df["X_pred"]
timed_comparison_df["Y_pred_m_KNN"] = timed_comparison_df["Y_pred"]

# Define methods for each value of k
methods = []
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for idx, k in enumerate(range(3, 7)):
    df_k = timed_comparison_df[timed_comparison_df["k"] == k].copy().reset_index(drop=True)
    methods.append((f"k={k}", colors[idx], df_k))

# Determine global min/max
x_all = np.concatenate([
    df["X_true_m"].values for _, _, df in methods
] + [
    df["X_pred_m_KNN"].values for _, _, df in methods
])
y_all = np.concatenate([
    df["Y_true_m"].values for _, _, df in methods
] + [
    df["Y_pred_m_KNN"].values for _, _, df in methods
])

# Axis margins
margin_x = 1.0
margin_y = 2.0
x_min = x_all.min() - margin_x
x_max = x_all.max() + margin_x
y_min = y_all.min() - margin_y
y_max = y_all.max() + margin_y

# Create the figure
fig, ax = plt.subplots(figsize=(3.8, 2.4), dpi=600)

# Plot ground truth
ax.scatter(methods[0][2]["X_true_m"], methods[0][2]["Y_true_m"], c='black', label='Ground Truth', marker='o', s=10)

# Plot each KNN variant
for label, color, df in methods:
    ax.scatter(df["X_pred_m_KNN"], df["Y_pred_m_KNN"], c=color, label=label, marker='^', s=10)
    for i in range(len(df)):
        ax.plot(
            [df.loc[i, "X_true_m"], df.loc[i, "X_pred_m_KNN"]],
            [df.loc[i, "Y_true_m"], df.loc[i, "Y_pred_m_KNN"]],
            color=color, linestyle='--', linewidth=0.4
        )

# Set limits and formatting
# Set limits and formatting
# ax.set_xlim(x_min, x_max)
# ax.set_ylim(y_min, y_max)
ax.set_xlabel("X (m)", fontsize=6)
ax.set_ylabel("Y (m)", fontsize=6)
ax.tick_params(axis='both', direction='in', length=2, width=0.3, 
               labelsize=6, top=True, right=True)
ax.set_aspect('equal', adjustable='box')  # ensures equal scaling
ax.grid(True, linestyle=':', linewidth=0.3)

# Legend
legend = ax.legend(fontsize=5, ncol=5, frameon=True)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.3)

# Layout and save
plt.tight_layout()
pdf_path = "knn-based_xy.pdf"
fig.savefig(pdf_path, dpi=1000, bbox_inches='tight', pad_inches=0.01, transparent=False)
plt.show()
