# Extracted provenance cell for tree-based_xy.pdf
# Source notebook: tree-based.ipynb
# Source cell index: 13
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt
import numpy as np

# Define your methods with (label, color, DataFrame)
methods = [
    ("RF", "#1f77b4", rf_df),
    ("XGB", "#2ca02c", xgb_df),
    ("CAT", "#ff7f0e", cat_df)
]

# Determine global min/max from all three methods
x_all = np.concatenate([
    rf_df["X_true_m"].values,
    rf_df["X_pred_m_RF"].values,
    xgb_df["X_pred_m_XGB"].values,
    cat_df["X_pred_m_CAT"].values
])

y_all = np.concatenate([
    rf_df["Y_true_m"].values,
    rf_df["Y_pred_m_RF"].values,
    xgb_df["Y_pred_m_XGB"].values,
    cat_df["Y_pred_m_CAT"].values
])

# Compute axis limits with margins
margin_x = 1.0  # meters
margin_y = 2.0  # meters

x_min = x_all.min() - margin_x
x_max = x_all.max() + margin_x
y_min = y_all.min() - margin_y
y_max = y_all.max() + margin_y

# Re-plot with margins
fig, ax = plt.subplots(figsize=(3.8, 2.7), dpi=600)

# Ground truth
ax.scatter(rf_df["X_true_m"], rf_df["Y_true_m"], c='black', label='Ground Truth', marker='o', s=10)

# Predictions with error lines
for label, color, df in methods:
    ax.scatter(df[f"X_pred_m_{label}"], df[f"Y_pred_m_{label}"], c=color, label=f"{label} Prediction", marker='^', s=10)
    for i in range(len(df)):
        ax.plot(
            [df.loc[i, "X_true_m"], df.loc[i, f"X_pred_m_{label}"]],
            [df.loc[i, "Y_true_m"], df.loc[i, f"Y_pred_m_{label}"]],
            color=color, linestyle='--', linewidth=0.4
        )

# Apply computed limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Axis formatting
ax.set_xlabel("X (m)", fontsize=7)
ax.set_ylabel("Y (m)", fontsize=7)
ax.tick_params(axis='both', direction='in', length=2, width=0.3, labelsize=6, top=True, right=True)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', linewidth=0.3)
legend = ax.legend(fontsize=5, ncol=4, frameon=True)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.3)
plt.tight_layout()
pdf_path = "tree-based_xy.pdf"
fig.savefig(pdf_path, dpi=1000, bbox_inches='tight', pad_inches=0.01, transparent=False)

plt.show()
