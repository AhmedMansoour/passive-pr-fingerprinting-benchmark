"""K-Weighted Nearest Neighbor (KWNN) localization models."""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def train_kwnn(X_train, y_train, k=6, metric="euclidean"):
    """Train a KWNN model with distance weighting."""
    model = KNeighborsRegressor(n_neighbors=k, weights="distance", metric=metric)
    model.fit(X_train, y_train)
    return model


def run_kwnn_grid(X_train, y_train, X_test, k_values, metrics):
    """
    Run KWNN across all k values and distance metrics.
    Returns list of (metric, k, y_pred) tuples.
    """
    results = []
    for metric in metrics:
        for k in k_values:
            model = train_kwnn(X_train, y_train, k=k, metric=metric)
            y_pred = model.predict(X_test)
            results.append({"metric": metric, "k": k, "y_pred": y_pred, "model": model})
    return results
