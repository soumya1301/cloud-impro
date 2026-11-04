"""
Sliding-window dataset generation for multivariate time-series forecasting.
Enforces strict entity isolation (no windows spanning different VMs or workloads).
"""
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesWindowDataset(Dataset):
    """PyTorch Dataset yielding (X, y) sliding windows for multivariate forecasting."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_sliding_windows(
    data: np.ndarray,
    history_len: int = 12,
    horizon: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs sliding windows from a single contiguous multivariate time series.
    Input data shape: (T, num_features)
    Output X shape: (N, history_len, num_features)
    Output y shape: (N, num_features) at time t + horizon
    """
    total_steps = len(data)
    X, y = [], []
    for i in range(total_steps - history_len - horizon + 1):
        X.append(data[i : i + history_len])
        y.append(data[i + history_len + horizon - 1])
    if len(X) == 0:
        return np.empty((0, history_len, data.shape[1] if data.ndim > 1 else 1)), np.empty((0, data.shape[1] if data.ndim > 1 else 1))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class VMAwareWindowGenerator:
    """
    Constructs sliding windows strictly partitioned by entity (VM or Workload Collection).
    Guarantees:
    1. Entity ID isolation: No window ever crosses entity boundaries.
    2. Chronological ordering: Observations within each entity are strictly sorted by timestamp.
    3. Horizon alignment: Target y is exactly at t + history_len + horizon - 1.
    4. Explicit tracking of eligible entities and dropped windows.
    """

    def __init__(
        self,
        history_len: int = 12,
        horizon: int = 2,
        feature_cols: Optional[List[str]] = None,
        entity_col: str = "vm_id",
        time_col: str = "start_time"
    ):
        self.history_len = history_len
        self.horizon = horizon
        self.feature_cols = feature_cols or ["cpu_util", "mem_util"]
        self.entity_col = entity_col
        self.time_col = time_col

    def generate_windows(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Extracts sliding windows across all entities in df.
        Returns:
            X: np.ndarray of shape (N, history_len, num_features)
            y: np.ndarray of shape (N, num_features)
            stats: Dict containing accounting metrics (total entities, eligible entities, total windows)
        """
        min_required_steps = self.history_len + self.horizon
        grouped = df.groupby(self.entity_col)
        total_entities = len(grouped)
        eligible_entities = 0
        total_windows = 0

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        entity_metadata: List[Dict[str, Any]] = []

        for entity_id, group in grouped:
            if len(group) < min_required_steps:
                continue

            # Sort strictly chronologically
            sorted_group = group.sort_values(self.time_col)
            features = sorted_group[self.feature_cols].values.astype(np.float32)
            n_steps = len(features)

            eligible_entities += 1
            n_windows = n_steps - min_required_steps + 1
            total_windows += n_windows

            for i in range(n_windows):
                X_list.append(features[i : i + self.history_len])
                y_list.append(features[i + self.history_len + self.horizon - 1])

        num_features = len(self.feature_cols)
        if X_list:
            X_arr = np.array(X_list, dtype=np.float32)
            y_arr = np.array(y_list, dtype=np.float32)
        else:
            X_arr = np.empty((0, self.history_len, num_features), dtype=np.float32)
            y_arr = np.empty((0, num_features), dtype=np.float32)

        stats = {
            "entity_col": self.entity_col,
            "history_len": self.history_len,
            "horizon": self.horizon,
            "min_required_steps": min_required_steps,
            "total_entities_inspected": total_entities,
            "eligible_entities": eligible_entities,
            "ineligible_entities": total_entities - eligible_entities,
            "total_windows_generated": len(X_arr)
        }

        return X_arr, y_arr, stats


def create_vm_aware_sliding_windows(
    df: pd.DataFrame,
    entity_col: str = "vm_id",
    time_col: str = "start_time",
    feature_cols: Optional[List[str]] = None,
    history_len: int = 12,
    horizon: int = 2
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience function for VMAwareWindowGenerator."""
    generator = VMAwareWindowGenerator(
        history_len=history_len,
        horizon=horizon,
        feature_cols=feature_cols,
        entity_col=entity_col,
        time_col=time_col
    )
    return generator.generate_windows(df)
