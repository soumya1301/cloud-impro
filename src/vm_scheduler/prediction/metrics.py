"""
Evaluation metrics for resource demand prediction: MAE, RMSE, MAPE, R2, and NMAE.
Includes bounded MAPE to handle near-zero target values typical of Borg cluster traces.
"""
from typing import Dict, Any
import numpy as np


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 0.01
) -> Dict[str, float]:
    """
    Calculates regression metrics across predicted resource dimensions.
    
    Provides:
    - mae: Mean Absolute Error (physical unscaled units)
    - rmse: Root Mean Squared Error (physical unscaled units)
    - r2: Coefficient of determination
    - nmae: Normalized MAE (MAE / range(y_true))
    - mape_raw: Standard MAPE with epsilon clipping (inflated by near-zero denominators)
    - mape_bounded: MAPE computed strictly on targets >= zero_threshold
    """
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Raw MAPE (subject to near-zero inflation)
    eps = 1e-4
    mape_raw = float(np.mean(np.abs(diff) / np.maximum(np.abs(y_true), eps)) * 100.0)

    # Bounded MAPE: filter targets with active non-negligible utilization
    non_zero_mask = np.abs(y_true) >= zero_threshold
    if np.any(non_zero_mask):
        mape_bounded = float(np.mean(np.abs(diff[non_zero_mask]) / np.abs(y_true[non_zero_mask])) * 100.0)
    else:
        mape_bounded = mape_raw

    # Normalized MAE
    val_range = float(np.max(y_true) - np.min(y_true))
    nmae = float(mae / max(val_range, 1e-6))

    # R2 Score
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "nmae": round(nmae, 5),
        "mape": round(mape_raw, 2),
        "mape_bounded": round(mape_bounded, 2),
        "r2": round(r2, 4)
    }
