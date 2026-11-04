"""
Normalization and scaling utilities for multivariate resource time series.
"""
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np


class ResourceScaler:
    """Min-Max scaler preserving lower and upper capacity bounds with state persistence."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range
        self.min_vals: Optional[np.ndarray] = None
        self.max_vals: Optional[np.ndarray] = None
        self._diff: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ResourceScaler":
        """Calculates per-feature minimum and maximum across 2D (N, D) or 3D (N, T, D) arrays."""
        if X.ndim == 3:
            # Flatten across batch and time dimensions for fitting
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        self.min_vals = np.nanmin(X_flat, axis=0)
        self.max_vals = np.nanmax(X_flat, axis=0)
        # Avoid division by zero for constant features
        diff = self.max_vals - self.min_vals
        diff[diff == 0] = 1.0
        self._diff = diff
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scales input array into target range (supports 2D and 3D shapes)."""
        if self.min_vals is None or self._diff is None:
            raise ValueError("Scaler has not been fitted.")

        orig_shape = X.shape
        if X.ndim == 3:
            X_flat = X.reshape(-1, orig_shape[-1])
            scaled = (X_flat - self.min_vals) / self._diff
            scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
            return scaled.reshape(orig_shape).astype(np.float32)
        else:
            scaled = (X - self.min_vals) / self._diff
            return (scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits scaler and transforms input array."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverts scaled array back to original units (supports 2D and 3D shapes)."""
        if self.min_vals is None or self._diff is None:
            raise ValueError("Scaler has not been fitted.")

        orig_shape = X.shape
        if X.ndim == 3:
            X_flat = X.reshape(-1, orig_shape[-1])
            unscaled = (X_flat - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            unscaled = unscaled * self._diff + self.min_vals
            return unscaled.reshape(orig_shape).astype(np.float32)
        else:
            unscaled = (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
            return (unscaled * self._diff + self.min_vals).astype(np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Serializes scaler state for checkpoint bundling."""
        if self.min_vals is None:
            return {}
        return {
            "feature_range": self.feature_range,
            "min_vals": self.min_vals.tolist(),
            "max_vals": self.max_vals.tolist(),
            "diff": self._diff.tolist() if self._diff is not None else None
        }

    def set_state(self, state: Dict[str, Any]) -> "ResourceScaler":
        """Restores scaler state from checkpoint."""
        if not state:
            return self
        self.feature_range = tuple(state["feature_range"])
        self.min_vals = np.array(state["min_vals"], dtype=np.float32)
        self.max_vals = np.array(state["max_vals"], dtype=np.float32)
        if "diff" in state and state["diff"] is not None:
            self._diff = np.array(state["diff"], dtype=np.float32)
        else:
            diff = self.max_vals - self.min_vals
            diff[diff == 0] = 1.0
            self._diff = diff
        return self

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ResourceScaler":
        scaler = cls()
        scaler.set_state(state)
        return scaler
