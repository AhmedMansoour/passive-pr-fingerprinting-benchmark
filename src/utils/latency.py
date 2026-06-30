"""
Latency measurement protocol for inference timing.
Uses single-sample prediction with warmup and high-precision timer.
"""
import time
import numpy as np


def measure_latency(model, test_input, num_repetitions=100, warmup_reps=10):
    """Measure single-sample inference latency with proper protocol.

    Args:
        model: Trained model with .predict() method.
        test_input: Single test sample, shape (1, n_features).
        num_repetitions: Number of timed repetitions.
        warmup_reps: Number of untimed warmup calls.

    Returns:
        dict with mean_ms, median_ms, p95_ms, p99_ms, min_ms, max_ms, std_ms.
    """
    # Warmup phase
    for _ in range(warmup_reps):
        model.predict(test_input)

    # Measurement phase
    latencies = np.empty(num_repetitions)
    for i in range(num_repetitions):
        start = time.perf_counter()
        model.predict(test_input)
        end = time.perf_counter()
        latencies[i] = (end - start) * 1000.0  # ms

    return {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies),
        "std_ms": np.std(latencies),
    }
