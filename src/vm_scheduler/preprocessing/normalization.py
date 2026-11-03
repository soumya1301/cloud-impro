"""
Normalization and scaling utilities for multivariate resource time series.
"""
from typing import Tuple, Optional
import numpy as np


class ResourceScaler:
    """Min-Max scaler preserving lower and upper capacity bounds."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ResourceScaler":
        """Calculates per-feature minimum and maximum."""
        self.min_vals = np.nanmin(X, axis=0)
        self.max_vals = np.nanmax(X, axis=0)
        # Avoid division by zero for constant features
        diff = self.max_vals - self.min_vals
        diff[diff == 0] = 1.0
        self._diff = diff
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales input array into target range."""
        if self.min_vals is None:
            raise ValueError("Scaler has not been fitted.")
        scaled = (X - self.min_vals) / self._diff
        return scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits scaler and transforms input array."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverts scaled array back to original units."""
        if self.min_vals is None:
            raise ValueError("Scaler has not been fitted.")
        unscaled = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return unscaled * self._diff + self.min_vals

