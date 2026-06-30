# Extracted provenance cell for ann_architectures_xy_subset.pdf
# Source notebook: nn_senstivity.ipynb
# Source cell index: 9
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# ─────────────────────────────────────────────────────────────
# Compact plotting (Elsevier single-column size)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.6, 2.4), dpi=600)



# Predictions + dashed error lines
for label, color, df in methods:
    ax.scatter(df["X_pred"], df["Y_pred"],
               c=color, marker="^", s=8, label=f"{label}")
    for i in range(len(df)):
        ax.plot([df.loc[i, "X_actual"], df.loc[i, "X_pred"]],
                [df.loc[i, "Y_actual"], df.loc[i, "Y_pred"]],
                linestyle="--", linewidth=0.35, color=color)
# Ground truth
ax.scatter(arch_dfs[1]["X_actual"], arch_dfs[1]["Y_actual"],
           c="black", marker="o", s=8, label="Ground Truth")
# Axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Labels and aspect ratio
ax.set_xlabel("X (m)", fontsize=6)
ax.set_ylabel("Y (m)", fontsize=6)
ax.set_aspect("equal", adjustable="box")

# Ticks inside on all sides
ax.tick_params(axis="both", direction="in", length=2, width=0.3,
               labelsize=5, top=True, right=True)

# Subtle grid
ax.grid(True, linestyle=":", linewidth=0.25, alpha=0.6)

# Compact legend
legend = ax.legend(fontsize=5, ncol=5, loc="upper center",
                   frameon=True, bbox_to_anchor=(0.5, 1.03))
legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.3)

plt.tight_layout(pad=0.5)
pdf_path = "ann_architectures_xy_subset.pdf"
fig.savefig(pdf_path, dpi=1000, bbox_inches="tight", pad_inches=0.01)
print(f"✓ Figure saved to {pdf_path}")
plt.show()
