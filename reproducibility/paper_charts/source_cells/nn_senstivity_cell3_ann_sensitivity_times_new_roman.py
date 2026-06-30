# Extracted provenance cell for ann_sensitivity_times_new_roman.png
# Source notebook: nn_senstivity.ipynb
# Source cell index: 3
# Note: This is the original plotting cell as provided. Some cells rely on
# upstream notebook variables; adapted runnable scripts are placed in ../adapted_scripts.

import matplotlib.pyplot as plt

# Use Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'

# Common epochs
epochs = [
    50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
    550, 600, 650, 700, 750, 800, 850, 900, 950, 1000
]

# Continuous naming (Arch 1–4)
arch_data = {
    "Arch 1": {
        "rmse": [10304.27117, 5599.072495, 5437.290369, 5434.658564, 5218.593002, 5068.332365,
                 4885.458417, 4573.263547, 3808.598927, 3204.486981, 3132.839014, 2508.072163,
                 2228.660687, 2185.354742, 2211.885568, 2386.639352, 2204.083837, 2331.218110,
                 2185.331253, 2225.634293],
        "time": [0.151654005, 0.127846003, 0.149515867, 0.138581991, 0.136605024, 0.148810148,
                 0.148406267, 0.146022558, 0.149690866, 0.155007839, 0.169167995, 0.165140629,
                 0.150636196, 0.150125504, 0.152487278, 0.143482208, 0.159204960, 0.134609461,
                 0.141377211, 0.146171570]
    },
    "Arch 2": {
        "rmse": [5970.646428, 5394.979632, 5237.360481, 4569.870799, 3445.621809, 2469.887707,
                 2222.581621, 2227.553168, 2683.472624, 2188.241311, 2227.244933, 2196.058894,
                 2089.554482, 2393.463885, 2190.255053, 2251.444253, 2138.754599, 2305.960986,
                 2291.053553, 2159.125264],
        "time": [0.161722660, 0.165079594, 0.166412592, 0.162617922, 0.157900333, 0.166344643,
                 0.167447567, 0.177322865, 0.169112682, 0.159701347, 0.165897608, 0.163278103,
                 0.162462473, 0.162505627, 0.174988270, 0.170946836, 0.153694630, 0.164552689,
                 0.157557964, 0.164723158]
    },
    "Arch 3": {  # was Arch 4
        "rmse": [8243.082215, 5476.880632, 5162.982727, 4659.829582, 3868.992018, 2266.265206,
                 2185.825510, 2201.317968, 2234.185534, 2136.483190, 2217.221867, 2107.623421,
                 2103.676854, 2165.069960, 2186.483345, 2247.941934, 2258.689774, 2224.218035,
                 2208.709254, 2268.351152],
        "time": [0.138870478, 0.134604454, 0.148905277, 0.149753809, 0.146317482, 0.146463156,
                 0.135100126, 0.149357319, 0.153004169, 0.150651455, 0.145185709, 0.143318892,
                 0.148812771, 0.146309853, 0.148178101, 0.148342609, 0.148742199, 0.145864487,
                 0.141361713, 0.146196842]
    },
    "Arch 4": {  # was Arch 6
        "rmse": [5603.535117, 5168.050245, 4853.224495, 3265.862462, 2808.931280, 2417.853840,
                 2306.465048, 2319.097876, 2245.415243, 2108.189604, 2102.626982, 2132.964349,
                 2189.387814, 2286.733795, 2164.191228, 2223.134944, 2169.004358, 2220.170466,
                 2205.986328, 2278.013786],
        "time": [0.146683455, 0.148337841, 0.138700962, 0.143267155, 0.152848005, 0.151099682,
                 0.150964022, 0.143783569, 0.148134470, 0.148918867, 0.159625530, 0.145416260,
                 0.157433510, 0.159515858, 0.151792288, 0.164081097, 0.147634268, 0.159569740,
                 0.145881414, 0.149405718]
    }
}

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# (a) RMSE in meters
for name, metrics in arch_data.items():
    rmse_m = [v / 1000.0 for v in metrics["rmse"]]  # mm → m
    ax1.plot(epochs, rmse_m, marker="o", label=name)
ax1.set_ylabel("RMSE (m)", fontsize=12)
ax1.set_title("(a) ANN Sensitivity: RMSE vs. Epochs", fontsize=14)
ax1.legend(loc="upper center", ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.9))
ax1.grid(True)
ax1.tick_params(axis='both', direction="in")

# (b) Prediction time in milliseconds
for name, metrics in arch_data.items():
    time_ms = [t * 1000.0 for t in metrics["time"]]  # s → ms
    ax2.plot(epochs, time_ms, marker="s", linestyle="--", label=name)
ax2.set_xlabel("Epochs", fontsize=12)
ax2.set_ylabel("Prediction Time (ms)", fontsize=12)
ax2.set_title("(b) ANN Sensitivity: Inference Latency vs. Epochs", fontsize=14)
ax2.legend(loc="upper center", ncol=4, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.18))
ax2.grid(True)
ax2.tick_params(axis='both', direction="in")

plt.tight_layout()
plt.savefig("ann_sensitivity_times_new_roman.png", dpi=500, bbox_inches="tight")
plt.show()
