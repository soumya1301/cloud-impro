"""
Evaluation metrics for resource demand prediction: MAE, RMSE, MAPE, and R2.
"""
from typing import Dict
import numpy as np


def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression metrics across all predicted resource dimensions."""
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Epsilon-clipped MAPE to prevent zero division
    eps = 1e-4
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), eps)) * 100.0)

    # R2 Score
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "mape": round(mape, 2),
        "r2": round(r2, 4)
    }

