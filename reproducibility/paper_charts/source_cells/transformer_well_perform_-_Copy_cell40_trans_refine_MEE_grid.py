# Extracted provenance cell for trans_refine_MEE_grid.pdf
# Source notebook: transformer_well_perform - Copy.ipynb
# Source cell index: 40
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# ============================================================
# AE+TF refinement plots: MEE vs Bottleneck size (B)
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# df_meta prepared earlier (with columns: B, lambda, noise, wd, loss, MEE)
os.makedirs("figures", exist_ok=True)

# --- Friendly formatting helpers ---
def format_wd(val: float) -> str:
    if np.isclose(val, 0.0):
        return "0"
    if np.isclose(val, 1e-5):
        return r"$10^{-5}$"
    # fallback (rare)
    return f"{val:g}"

def format_lambda(val: float) -> str:
    # render as LaTeX lambda
    return rf"$\lambda$={val:g}"

# Facet by (noise, wd, loss)
facet_keys = (
    df_meta[["noise", "wd", "loss"]]
    .drop_duplicates()
    .sort_values(["noise", "wd", "loss"])
    .values
    .tolist()
)

n_facets = len(facet_keys)
ncols = min(3, n_facets)
nrows = int(np.ceil(n_facets / ncols))

fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(3.6, 2.2 + 0.8*(nrows-1)), dpi=600
)
if nrows == 1 and ncols == 1:
    axes = np.array([[axes]])
elif nrows == 1:
    axes = np.array([axes])
axes = axes[:nrows, :ncols]

# Palette/markers per λ (repeat if more)
lambda_values_sorted = sorted(df_meta["lambda"].unique())
colors = plt.cm.tab10.colors
markers = ["o", "s", "^", "D", "v", "P", "X"]

for idx, (noise_v, wd_v, loss_v) in enumerate(facet_keys):
    r = idx // ncols
    c = idx % ncols
    ax = axes[r, c]

    sub = df_meta[
        (df_meta["noise"] == noise_v) &
        (df_meta["wd"] == wd_v) &
        (df_meta["loss"] == loss_v)
    ].copy()

    if sub.empty:
        ax.set_visible(False)
        continue

    # Plot one line per λ (sorted for consistency)
    for i, lam in enumerate(lambda_values_sorted):
        chunk = sub[sub["lambda"] == lam].sort_values("B")
        if chunk.empty:
            continue
        ax.plot(
            chunk["B"].values, chunk["MEE"].values,
            marker=markers[i % len(markers)],
            linewidth=1.0,
            label=format_lambda(lam),
            color=colors[i % len(colors)],
            markersize=1.5
        )

    # Cosmetics
    ax.set_title(
        rf"Noise={noise_v:g}, wd={format_wd(wd_v)}, loss={loss_v}",
        fontsize=4.5, pad=2
    )
    ax.set_xlabel("Bottleneck (B)", fontsize=5)
    ax.set_ylabel("MEE (m)", fontsize=5)
    ax.tick_params(
        which="major", direction="in",
        top=True, right=True,
        length=2, width=0.3, labelsize=5
    )
    ax.tick_params(
        which="minor", direction="in",
        top=True, right=True,
        length=1, width=0.2, labelsize=5
    )
    ax.xaxis.set_major_locator(MultipleLocator(4 if sub['B'].nunique() > 1 else 1))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(True, axis="y", which="both", linestyle="--", linewidth=0.3, alpha=0.35)

    if sub["lambda"].nunique() > 1:
        leg = ax.legend(fontsize=5, frameon=True)
        leg.get_frame().set_edgecolor('black')
        leg.get_frame().set_linewidth(0.2)

# Hide unused subplots
for k in range(idx + 1, nrows * ncols):
    r = k // ncols
    c = k % ncols
    axes[r, c].set_visible(False)

plt.tight_layout()
fig.savefig("trans_refine_MEE_grid.pdf", dpi=1000,
            bbox_inches='tight', pad_inches=0.01, transparent=False)
plt.show()
