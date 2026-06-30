"""Maximum A Posteriori (MAP) probabilistic localization."""

import numpy as np


class MAPEstimator:
    """
    MAP-based indoor localization using multivariate Gaussian likelihood.

    For each test point, computes the likelihood of the observed RSSI vector
    under each reference point's Gaussian model, then returns the weighted
    average of the top-k most likely positions.
    """

    def __init__(self, k=6):
        self.k = k
        self.means = None      # Per-point mean RSSI vectors
        self.variances = None   # Per-point diagonal variances
        self.coordinates = None # Per-point (X, Y) coordinates

    def fit(self, raw_df, coord_mapping, default_rssi=-100):
        """
        Fit the MAP model by computing per-point Gaussian parameters.

        Args:
            raw_df: Training DataFrame with [Point, AP, Signal] columns
            coord_mapping: Dict mapping point names to (X, Y)
            default_rssi: Default for missing APs
        """
        raw_df = raw_df.copy()
        raw_df["Point"] = raw_df["Point"].str.lower()

        points = sorted(raw_df["Point"].unique())
        aps = sorted(raw_df["AP"].unique())

        means_list = []
        vars_list = []
        coords_list = []

        for point in points:
            if point not in coord_mapping:
                continue
            point_data = raw_df[raw_df["Point"] == point]
            mean_vec = []
            var_vec = []
            for ap in aps:
                ap_signals = point_data[point_data["AP"] == ap]["Signal"].values
                if len(ap_signals) > 0:
                    mean_vec.append(ap_signals.mean())
                    var_vec.append(max(ap_signals.var(), 1.0))  # Floor variance
                else:
                    mean_vec.append(default_rssi)
                    var_vec.append(100.0)

            means_list.append(mean_vec)
            vars_list.append(var_vec)
            coords_list.append(coord_mapping[point])

        self.means = np.array(means_list)
        self.variances = np.array(vars_list)
        self.coordinates = np.array(coords_list)
        self.aps = aps

    def predict_single(self, rssi_vector):
        """Predict position for a single RSSI observation."""
        # Log-likelihood under diagonal Gaussian
        diff = rssi_vector - self.means
        log_likelihoods = -0.5 * np.sum(diff ** 2 / self.variances + np.log(self.variances), axis=1)

        # Top-k indices
        top_k_idx = np.argsort(log_likelihoods)[-self.k:]
        top_k_weights = np.exp(log_likelihoods[top_k_idx] - log_likelihoods[top_k_idx].max())
        top_k_weights /= top_k_weights.sum()

        # Weighted average of coordinates
        predicted = np.average(self.coordinates[top_k_idx], weights=top_k_weights, axis=0)
        return predicted

    def predict(self, X_test):
        """Predict positions for multiple test observations."""
        predictions = np.array([self.predict_single(x) for x in X_test])
        return predictions
