#!/usr/bin/env python
"""Generate ap_density_sensitivity.pdf from archived AP-density result CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FULL = ROOT / "results" / "ap_density_full.csv"
CORE = ROOT / "results" / "ap_density_core.csv"
OUT_DIR = ROOT / "paper_outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "ap_density_sensitivity.pdf"

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})

full_df = pd.read_csv(FULL)
core_df = pd.read_csv(CORE)
models = ["KWNN", "RF", "Transformer"]
colors = {"KWNN": "#1f77b4", "RF": "#2ca02c", "Transformer": "#d62728"}
markers = {"KWNN": "o", "RF": "s", "Transformer": "^"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

ax = axes[0]
for model_name in models:
    mdf = full_df[full_df["model"] == model_name]
    k_vals = sorted(mdf["k"].unique())
    means, mins, maxs = [], [], []
    for k in k_vals:
        kdf = mdf[mdf["k"] == k]
        means.append(kdf["Mean_Error_m"].mean())
        mins.append(kdf["Mean_Error_m"].min())
        maxs.append(kdf["Mean_Error_m"].max())
    means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
    ax.plot(k_vals, means, marker=markers[model_name], color=colors[model_name],
            label=model_name, linewidth=2, markersize=7)
    ax.fill_between(k_vals, mins, maxs, alpha=0.15, color=colors[model_name])
ax.set_xlabel("Number of Monitor Nodes ($k$)", fontsize=11)
ax.set_ylabel("Mean Localization Error (m)", fontsize=11)
ax.set_title("(a) Effect of Monitor-Node Density", fontsize=12)
ax.set_xticks([2, 3, 4, 5, 6])
ax.legend(fontsize=9)

ax = axes[1]
k4 = core_df[core_df["k"] == 4]
geo_order = ["clustered", "median", "dispersed"]
x = np.arange(len(geo_order)); width = 0.22
for i, model_name in enumerate(models):
    vals = []
    for geo in geo_order:
        row = k4[(k4["model"] == model_name) & (k4["geometry"] == geo)]
        vals.append(row["Mean_Error_m"].values[0] if len(row) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=model_name, color=colors[model_name],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x + width)
ax.set_xticklabels([g.capitalize() for g in geo_order], fontsize=10)
ax.set_ylabel("Mean Localization Error (m)", fontsize=11)
ax.set_title("(b) Effect of Deployment Geometry ($k=4$)", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", format="pdf")
print(f"Saved {OUT}")
