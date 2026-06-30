"""
HNSW-based approximate k-NN for WiFi fingerprint localization.
Uses hnswlib for fast approximate nearest neighbor retrieval.
"""
import numpy as np

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


def train_hnsw(X_train, y_train, k=5, ef_construction=200, M=16, ef_search=50):
    """Build an HNSW index and return a predictor function.

    Args:
        X_train: Training features, shape (n_samples, n_features).
        y_train: Training targets (coordinates), shape (n_samples, 2).
        k: Number of neighbors to retrieve.
        ef_construction: HNSW construction parameter (higher = more accurate, slower build).
        M: HNSW max connections per layer.
        ef_search: HNSW search parameter (higher = more accurate, slower query).

    Returns:
        dict with 'predict' callable function and 'index' object.
    """
    if not HAS_HNSWLIB:
        raise ImportError(
            "hnswlib is required for HNSW retrieval. Install with: pip install hnswlib"
        )

    n_samples, n_features = X_train.shape

    # Build index
    index = hnswlib.Index(space="l2", dim=n_features)
    index.init_index(max_elements=n_samples, ef_construction=ef_construction, M=M)
    index.add_items(X_train, np.arange(n_samples))
    index.set_ef(ef_search)

    def predict(X_test):
        """Predict coordinates using distance-weighted k-NN via HNSW.

        Args:
            X_test: Test features, shape (n_test, n_features).

        Returns:
            Predicted coordinates, shape (n_test, 2).
        """
        labels, distances = index.knn_query(X_test, k=k)
        # distances from hnswlib are squared L2
        distances = np.sqrt(distances + 1e-10)

        predictions = np.zeros((X_test.shape[0], 2))
        for i in range(X_test.shape[0]):
            weights = 1.0 / (distances[i] + 1e-10)
            weights /= weights.sum()
            predictions[i] = np.average(y_train[labels[i]], axis=0, weights=weights)

        return predictions

    return {"predict": predict, "index": index}


def run_hnsw_benchmark(X_train, y_train, X_test, y_test, k_values=None):
    """Run HNSW benchmark with multiple k values.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data with true coordinates.
        k_values: List of k values to test. Defaults to [3, 5, 7, 10].

    Returns:
        List of result dicts with method name and error statistics.
    """
    if k_values is None:
        k_values = [3, 5, 7, 10]

    results = []
    for k in k_values:
        model = train_hnsw(X_train, y_train, k=k)
        predictions = model["predict"](X_test)
        errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))

        results.append({
            "method": f"HNSW_k{k}",
            "k": k,
            "mean_error": errors.mean(),
            "median_error": np.median(errors),
            "max_error": errors.max(),
            "std_error": errors.std(),
            "errors": errors,
        })
        print(f"  HNSW k={k}: mean={errors.mean():.2f}, median={np.median(errors):.2f}")

    return results
