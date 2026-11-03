"""
Sliding-window dataset generation for multivariate time-series forecasting.
"""
from typing import Tuple, List
import numpy as np
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
    Constructs sliding windows from multi-variate time series.
    Input data shape: (T, num_features)
    Output X shape: (N, history_len, num_features)
    Output y shape: (N, num_features) at time t + horizon
    """
    total_steps = len(data)
    X, y = [], []
    for i in range(total_steps - history_len - horizon + 1):
        X.append(data[i : i + history_len])
        y.append(data[i + history_len + horizon - 1])
    return np.array(X), np.array(y)

