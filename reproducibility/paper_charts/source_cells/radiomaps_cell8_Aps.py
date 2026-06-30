# Extracted provenance cell for Aps.pdf
# Source notebook: radiomaps.ipynb
# Source cell index: 8
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plt.rcParams.update({
    'font.family': 'Times New Roman'
})
# Access point coordinates at z = 0
ap_coords = {
    "ap1": (7880, 26400),
    "ap2": (-600, 20400),
    "ap3": (1200, 13200),
    "ap4": (6080, 16850),
    "ap5": (-600, 1800),
    "ap6": (3600, 7800),
}

# Extract X, Y, Z
labels = list(ap_coords.keys())
y_vals = np.array([coord[0] for coord in ap_coords.values()])
x_vals = np.array([coord[1] for coord in ap_coords.values()])
z_vals = np.zeros_like(x_vals)  # All at z = 0
x_vals=x_vals/1000

y_vals=y_vals/1000
# Create 3D plot with compressed Z aspect
fig = plt.figure(figsize=(4, 3), dpi=1000)
ax = fig.add_subplot(111, projection='3d')

# Plot AP locations
ax.scatter(x_vals, y_vals, z_vals, color='blue', s=20)

# Annotate each AP
for label, x, y in zip(labels, x_vals, y_vals):
    ax.text(x+1.2, y+0.5, 0, f'{label.upper()}', fontsize=6, ha='center', va='bottom')

# Set the aspect ratio with small z-height
ax.set_box_aspect((np.ptp(x_vals), np.ptp(y_vals), 0.02 * np.ptp(y_vals)))  # Very flat Z

# Axis formatting
ax.set_xlabel('X (m)', fontsize=8, fontweight='bold', labelpad=+12)
ax.set_ylabel('Y (m)', fontsize=8, fontweight='bold', labelpad=-12)
ax.set_zlabel('')  # No z-axis label
ax.tick_params(axis='x', labelsize=6, width=0.3, length=1, pad=+5)
ax.tick_params(axis='y', labelsize=6, width=0.3, length=3, pad=-5)
ax.tick_params(axis='z', labelsize=6, width=0.3, length=3, pad=-3)
ax.xaxis._axinfo['grid'].update(color='gray', linestyle=':', linewidth=0.2)
ax.yaxis._axinfo['grid'].update(color='gray', linestyle=':', linewidth=0.2)
ax.set_zticks([])  # Hide z-axis ticks

# Optional: Add a margin for better visualization
x_margin = (x_vals.max() - x_vals.min()) * 0.05
y_margin = (y_vals.max() - y_vals.min()) * 0.1

# Create plot

plt.xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
plt.ylim(y_vals.min() - y_margin, y_vals.max() + y_margin)
ax.invert_yaxis()
ax.invert_xaxis()

# View
ax.view_init(elev=35, azim=50)

plt.tight_layout()
plt.savefig("Aps.pdf", dpi=1000,bbox_inches='tight', pad_inches=0.1)

plt.show()
