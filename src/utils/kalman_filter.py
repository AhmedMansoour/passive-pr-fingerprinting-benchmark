"""
1D Kalman filter for RSSI signal smoothing.
Applied per-AP to reduce noise in WiFi fingerprints.
"""
import numpy as np


class KalmanFilter1D:
    """Simple 1D Kalman filter for scalar time-series smoothing."""

    def __init__(self, process_variance=1e-3, measurement_variance=0.5):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def smooth(self, measurements):
        """Apply Kalman smoothing to a sequence of measurements.

        Args:
            measurements: 1D array of noisy measurements.

        Returns:
            Smoothed array of the same length.
        """
        n = len(measurements)
        if n == 0:
            return np.array([])

        smoothed = np.empty(n)

        # Initialize with first measurement
        estimate = measurements[0]
        error_estimate = 1.0

        for i in range(n):
            # Prediction
            error_estimate += self.process_variance

            # Update
            kalman_gain = error_estimate / (error_estimate + self.measurement_variance)
            estimate = estimate + kalman_gain * (measurements[i] - estimate)
            error_estimate = (1 - kalman_gain) * error_estimate

            smoothed[i] = estimate

        return smoothed
