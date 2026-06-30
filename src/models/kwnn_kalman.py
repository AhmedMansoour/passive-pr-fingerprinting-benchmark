"""Kalman-smoothed KWNN localization."""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


class KalmanFilter1D:
    """Simple 1D Kalman filter for RSSI smoothing."""

    def __init__(self, process_variance=1e-3, measurement_variance=0.5):
        self.Q = process_variance
        self.R = measurement_variance

    def smooth(self, measurements):
        """Apply Kalman smoothing to a sequence of RSSI measurements."""
        n = len(measurements)
        if n == 0:
            return measurements

        # Initialize
        x_hat = np.zeros(n)
        P = np.zeros(n)
        x_hat[0] = measurements[0]
        P[0] = 1.0

        # Forward pass
        for t in range(1, n):
            # Predict
            x_pred = x_hat[t - 1]
            P_pred = P[t - 1] + self.Q

            # Update
            K = P_pred / (P_pred + self.R)
            x_hat[t] = x_pred + K * (measurements[t] - x_pred)
            P[t] = (1 - K) * P_pred

        return x_hat


def build_kalman_fingerprints(raw_df, coord_mapping, default_rssi=-100,
                              process_variance=1e-3, measurement_variance=0.5):
    """
    Build fingerprint matrix with Kalman-smoothed RSSI per AP.

    Args:
        raw_df: Raw probe request DataFrame with columns [Point/Folder, AP, Signal, ...]
        coord_mapping: Dict mapping point names to (X, Y) coordinates
        default_rssi: Default RSSI for missing APs
        process_variance: Kalman Q parameter
        measurement_variance: Kalman R parameter

    Returns:
        Feature matrix X, coordinate matrix y, AP column names
    """
    kf = KalmanFilter1D(process_variance, measurement_variance)

    # Determine point column name
    point_col = "Point" if "Point" in raw_df.columns else "Folder"
    raw_df = raw_df.copy()
    raw_df[point_col] = raw_df[point_col].str.lower()

    smoothed_records = []
    for point in raw_df[point_col].unique():
        point_data = raw_df[raw_df[point_col] == point]
        for ap in point_data["AP"].unique():
            signals = point_data[point_data["AP"] == ap]["Signal"].values
            smoothed = kf.smooth(signals)
            smoothed_records.append({
                point_col: point,
                "AP": ap,
                "Signal": smoothed[-1]  # Use final smoothed value
            })

    smoothed_df = pd.DataFrame(smoothed_records)
    pivot = smoothed_df.pivot(index=point_col, columns="AP", values="Signal").fillna(default_rssi).reset_index()

    pivot["X"] = pivot[point_col].apply(lambda p: coord_mapping.get(p, (np.nan, np.nan))[0])
    pivot["Y"] = pivot[point_col].apply(lambda p: coord_mapping.get(p, (np.nan, np.nan))[1])
    pivot = pivot.dropna()

    ap_columns = sorted([col for col in pivot.columns if str(col).lower().startswith("ap")])
    X = pivot[ap_columns].values
    y = pivot[["X", "Y"]].values

    return X, y, ap_columns


def train_kalman_kwnn(X_train, y_train, k=6, metric="euclidean"):
    """Train KWNN on Kalman-smoothed fingerprints."""
    model = KNeighborsRegressor(n_neighbors=k, weights="distance", metric=metric)
    model.fit(X_train, y_train)
    return model
