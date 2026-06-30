# Extracted provenance cell for arch3_arch4_xy_meters_transformer.pdf
# Source notebook: best transformer than kwnn.ipynb
# Source cell index: 9
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt
import numpy as np

# Collect the two architectures’ per-point DataFrames
arch_datasets = [
    ("Arch 3", "#2ca02c", all_stage_perpoint["Arch 3 — AE+TF (λ=0, σ=0)"]),
    ("Arch 4", "#d62728", all_stage_perpoint["Arch 4 — AE+TF (λ=0.5, σ=0.1)"]),
]

# Convert mm → m
for _, _, df in arch_datasets:
    df["true_X_m"] = df["true_X_mm"] / 1000.0
    df["true_Y_m"] = df["true_Y_mm"] / 1000.0
    df["pred_X_m"] = df["pred_X_mm"] / 1000.0
    df["pred_Y_m"] = df["pred_Y_mm"] / 1000.0

# Gather all coords for shared limits
x_all = np.concatenate([df["true_X_m"].values for _, _, df in arch_datasets] +
                       [df["pred_X_m"].values for _, _, df in arch_datasets])
y_all = np.concatenate([df["true_Y_m"].values for _, _, df in arch_datasets] +
                       [df["pred_Y_m"].values for _, _, df in arch_datasets])

margin = 1.0
x_min, x_max = x_all.min() - margin, x_all.max() + margin
y_min, y_max = y_all.min() - margin, y_all.max() + margin

# ---- Plot both Arch 3 and Arch 4 ----
fig, ax = plt.subplots(figsize=(3.9, 2.4), dpi=600)

# Ground truth
ax.scatter(arch_datasets[0][2]["true_X_m"], arch_datasets[0][2]["true_Y_m"],
           c='black', label='Ground Truth', marker='o', s=10)

# Predictions + error vectors for both
for label, color, df in arch_datasets:
    ax.scatter(df["pred_X_m"], df["pred_Y_m"],
               c=color, label=label, marker='^', s=10)
    for i in range(len(df)):
        ax.plot(
            [df.loc[i, "true_X_m"], df.loc[i, "pred_X_m"]],
            [df.loc[i, "true_Y_m"], df.loc[i, "pred_Y_m"]],
            color=color, linestyle='--', linewidth=0.4
        )

# Axis: FIXED SCALE (1m in X = 1m in Y)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect("equal", adjustable="box")  # preserves scale ratio

# Labels & formatting
ax.set_xlabel("X (m)", fontsize=5)
ax.set_ylabel("Y (m)", fontsize=5)
ax.tick_params(axis='both', direction='in', length=2, width=0.3,
               labelsize=6, top=True, right=True)
ax.grid(True, linestyle=':', linewidth=0.3)

# Legend
legend = ax.legend(fontsize=5, ncol=3, frameon=True)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.3)

# Save
plt.tight_layout()
fig.savefig("arch3_arch4_xy_meters_transformer.pdf", dpi=1000,
            bbox_inches='tight', pad_inches=0.01, transparent=False)
plt.show()
