# Extracted provenance cell for hnsw_k_accuracy_trend.pdf
# Source notebook: Untitled1.ipynb
# Source cell index: 6
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

# Use k as x-axis and plot Recall vs k directly
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=500)

ax.plot(df_hnsw['k'], df_hnsw['Recall'], 'r-o', label='HNSW $k$ Sweep', linewidth=1.0, markersize=3)
best_k = df_hnsw[df_hnsw['Recall'] == df_hnsw['Recall'].max()]
ax.plot(best_k['k'], best_k['Recall'], 'ro', markersize=5)
ax.annotate('Optimal $k$', xy=(best_k['k'].values[0], best_k['Recall'].values[0]),
            xytext=(5, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', lw=0.5), fontsize=6)

# Formatting
ax.set_xlabel("Number of Neighbors ($k$)", fontsize=7)
ax.set_ylabel("Recall (1 / Error)", fontsize=7)
ax.set_title("HNSW Accuracy Trend Across $k$", fontsize=8)
ax.set_ylim(0.3, 0.55)
ax.set_xticks(df_hnsw['k'])
ax.grid(True, which='both', linestyle=':', linewidth=0.3)
ax.tick_params(axis='both', direction='in', length=2.5, width=0.4, labelsize=6)
legend = ax.legend(fontsize=5,  frameon=True)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.3)

plt.tight_layout()
plt.savefig("hnsw_k_accuracy_trend.pdf", format='pdf', dpi=600, bbox_inches='tight')

plt.show()
