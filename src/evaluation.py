"""Shared evaluation metrics for indoor localization."""

import numpy as np
import pandas as pd


def compute_euclidean_errors(y_true, y_pred):
    """Compute per-point Euclidean errors in millimeters."""
    error_x = np.abs(y_true[:, 0] - y_pred[:, 0])
    error_y = np.abs(y_true[:, 1] - y_pred[:, 1])
    return np.sqrt(error_x ** 2 + error_y ** 2)


def compute_error_statistics(errors_mm):
    """Compute summary statistics from errors in mm. Returns dict with values in meters."""
    errors_m = errors_mm / 1000.0
    return {
        "Mean_Error_m": errors_m.mean(),
        "Median_Error_m": np.median(errors_m),
        "Std_Error_m": errors_m.std(),
        "Min_Error_m": errors_m.min(),
        "Max_Error_m": errors_m.max(),
        "Q25_Error_m": np.percentile(errors_m, 25),
        "Q75_Error_m": np.percentile(errors_m, 75),
    }


def build_result_dataframe(test_pivot, y_pred, errors_mm, method_name, extra_cols=None):
    """Build a per-point result DataFrame."""
    result = test_pivot.copy()
    result["X_pred"] = y_pred[:, 0] / 1000.0
    result["Y_pred"] = y_pred[:, 1] / 1000.0
    result["X"] = result["X"] / 1000.0
    result["Y"] = result["Y"] / 1000.0
    result["Error_Euclidean"] = errors_mm / 1000.0
    result["Method"] = method_name
    if extra_cols:
        for k, v in extra_cols.items():
            result[k] = v
    return result


def print_summary(method_name, stats):
    """Print formatted summary."""
    print(f"  {method_name}: Mean={stats['Mean_Error_m']:.3f}m, "
          f"Median={stats['Median_Error_m']:.3f}m, "
          f"Max={stats['Max_Error_m']:.3f}m")
