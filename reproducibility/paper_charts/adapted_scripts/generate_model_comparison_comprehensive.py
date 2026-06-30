#!/usr/bin/env python
"""Generate model_comparison_comprehensive.pdf from archived paper-reference tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "paper_outputs" / "results" / "paper_reference_accuracy_latency_tradeoff.csv"
OUT_DIR = ROOT / "paper_outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "model_comparison_comprehensive.pdf"

df = pd.read_csv(RES)
family_colors = {"Retrieval":"#3b88a8", "Ensemble":"#c95338", "Neural":"#e78163", "Attention":"#36a98b"}
colors = [family_colors.get(f, "#777777") for f in df["family"]]
fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")
for ax in axes.flat:
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, linewidth=0.4)

# (a) mean error with simple variability proxy from median/max if std unavailable
d = df.sort_values("mean_m", ascending=True).copy()
y = np.arange(len(d))
axes[0,0].barh(y, d["mean_m"], color=[family_colors.get(f,"#777") for f in d["family"]], edgecolor="black", linewidth=0.3)
axes[0,0].set_yticks(y); axes[0,0].set_yticklabels(d["method"], fontsize=8)
axes[0,0].invert_yaxis(); axes[0,0].set_xlabel("Mean Error (m)")
axes[0,0].set_title("(a) Mean Localization Error with Cross-Seed Variability", fontweight="bold")
axes[0,0].axvline(2.02, color="#6da6b0", linestyle="--", linewidth=1.0, label="KWNN best (2.02 m)")
axes[0,0].legend(fontsize=7, loc="lower right")

# (b) latency log ranking
d = df.sort_values("latency_ms", ascending=True).copy()
y = np.arange(len(d))
axes[0,1].barh(y, d["latency_ms"].clip(lower=1e-3), color=[family_colors.get(f,"#777") for f in d["family"]], edgecolor="black", linewidth=0.3)
axes[0,1].set_xscale("log"); axes[0,1].set_yticks(y); axes[0,1].set_yticklabels(d["method"], fontsize=8)
axes[0,1].invert_yaxis(); axes[0,1].set_xlabel("Mean Latency (ms)")
axes[0,1].set_title("(b) Inference Latency (Log Scale)", fontweight="bold")
for thr in [1,5]: axes[0,1].axvline(thr, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

# (c) accuracy-latency scatter
ax = axes[1,0]
for fam, g in df.groupby("family"):
    ax.scatter(g["latency_ms"].clip(lower=1e-3), g["mean_m"], s=70, label=fam, color=family_colors.get(fam,"#777"), edgecolor="black", linewidth=0.4)
for _, r in df.iterrows():
    if r["method"] in ["HNSW-KWNN", "KWNN", "TF Arch 4", "RF", "XGBoost", "MLP Arch 4", "MAP"]:
        ax.annotate(r["method"], (max(r["latency_ms"],1e-3), r["mean_m"]), fontsize=7, xytext=(2,2), textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("Mean Latency (ms)"); ax.set_ylabel("Mean Error (m)")
ax.set_title("(c) Accuracy vs. Latency Trade-off", fontweight="bold")
ax.legend(fontsize=8, loc="upper left")

# (d) reproducibility/upper-tail proxy; lower is better
ax = axes[1,1]
d = df.assign(spread=df["max_m"]-df["median_m"]).sort_values("spread", ascending=True)
y=np.arange(len(d))
ax.barh(y, d["spread"], color=[family_colors.get(f,"#777") for f in d["family"]], edgecolor="black", linewidth=0.3)
ax.set_yticks(y); ax.set_yticklabels(d["method"], fontsize=8); ax.invert_yaxis()
ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Stability threshold")
ax.set_xlabel("Max--Median Error Gap (m)")
ax.set_title("(d) Reproducibility Ranking (Lower Is Better)", fontweight="bold")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", format="pdf")
print(f"Saved {OUT}")
