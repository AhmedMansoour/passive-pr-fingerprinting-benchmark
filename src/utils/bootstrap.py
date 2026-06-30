"""
Bootstrap confidence interval computation for localization error analysis.
"""
import numpy as np


def bootstrap_ci(errors, num_samples=10000, confidence_level=0.95, random_seed=42):
    """Compute bootstrap confidence intervals for error statistics.

    Args:
        errors: Array of error values (in any unit).
        num_samples: Number of bootstrap resamples.
        confidence_level: CI level (e.g. 0.95 for 95%).
        random_seed: Seed for reproducibility.

    Returns:
        dict with keys: mean, median, std, and their CI bounds.
    """
    alpha = 1 - confidence_level
    rng = np.random.RandomState(random_seed)

    bootstrap_means = np.empty(num_samples)
    bootstrap_medians = np.empty(num_samples)
    bootstrap_stds = np.empty(num_samples)

    n = len(errors)
    for i in range(num_samples):
        sample = rng.choice(errors, size=n, replace=True)
        bootstrap_means[i] = sample.mean()
        bootstrap_medians[i] = np.median(sample)
        bootstrap_stds[i] = sample.std()

    return {
        "mean": errors.mean(),
        "mean_ci_lower": np.percentile(bootstrap_means, (alpha / 2) * 100),
        "mean_ci_upper": np.percentile(bootstrap_means, (1 - alpha / 2) * 100),
        "median": np.median(errors),
        "median_ci_lower": np.percentile(bootstrap_medians, (alpha / 2) * 100),
        "median_ci_upper": np.percentile(bootstrap_medians, (1 - alpha / 2) * 100),
        "std": errors.std(),
        "std_ci_lower": np.percentile(bootstrap_stds, (alpha / 2) * 100),
        "std_ci_upper": np.percentile(bootstrap_stds, (1 - alpha / 2) * 100),
    }


def print_bootstrap_summary(name, ci):
    """Pretty-print bootstrap CI results."""
    print(f"  {name}:")
    print(f"    Mean:   {ci['mean']:.4f} [{ci['mean_ci_lower']:.4f}, {ci['mean_ci_upper']:.4f}]")
    print(f"    Median: {ci['median']:.4f} [{ci['median_ci_lower']:.4f}, {ci['median_ci_upper']:.4f}]")
    print(f"    Std:    {ci['std']:.4f} [{ci['std_ci_lower']:.4f}, {ci['std_ci_upper']:.4f}]")
