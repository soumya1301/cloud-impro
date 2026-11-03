"""
Time-series forecasting baselines: Persistence (Last Value) and Moving Average.
"""
from typing import Tuple
import numpy as np


class PersistenceForecaster:
    """Persistence baseline: predicts future value as the most recent observed value."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Takes input shape (N, seq_len, num_features) and returns the last time step (N, num_features)."""
        return X[:, -1, :]


class MovingAverageForecaster:
    """Moving average baseline: predicts future value as the average of recent observations."""

    def __init__(self, window: int = 6):
        self.window = window

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Computes mean over the last `window` time steps."""
        sub = X[:, -self.window:, :]
        return np.mean(sub, axis=1)

