"""
Multivariate LSTM training and evaluation script across 10m, 30m, and 60m forecast horizons.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.vm_scheduler.prediction.lstm import MultivariateLSTM
from src.vm_scheduler.prediction.trainer import LSTMTrainer
from src.vm_scheduler.prediction.baselines import PersistenceForecaster, MovingAverageForecaster
from src.vm_scheduler.prediction.metrics import compute_prediction_metrics
from src.vm_scheduler.preprocessing.windowing import create_sliding_windows, TimeSeriesWindowDataset
from src.vm_scheduler.preprocessing.normalization import ResourceScaler
from src.vm_scheduler.utils.config import setup_logger, set_seed, load_yaml_config

logger = setup_logger("train_lstm")


def train_and_evaluate_horizon(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    horizon_name: str,
    horizon_steps: int,
    cfg: dict
):
    logger.info(f"--- Training LSTM for Horizon: {horizon_name} ({horizon_steps * 5} min ahead) ---")

    train_dataset = TimeSeriesWindowDataset(X_train, y_train)
    val_dataset = TimeSeriesWindowDataset(X_val, y_val)
    test_dataset = TimeSeriesWindowDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = MultivariateLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"]
    )

    ckpt_path = f"models/checkpoints/lstm_{horizon_name}.pt"
    trainer = LSTMTrainer(
        model=model,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        device=cfg["training"]["device"]
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["early_stopping_patience"],
        checkpoint_path=ckpt_path
    )

    # Evaluate LSTM on held-out test set
    lstm_preds, true_vals, lstm_metrics = trainer.evaluate(test_loader)

    # Evaluate Baselines on the exact same test set
    pers_forecaster = PersistenceForecaster()
    pers_preds = pers_forecaster.predict(X_test)
    pers_metrics = compute_prediction_metrics(y_test, pers_preds)

    ma_forecaster = MovingAverageForecaster(window=6)
    ma_preds = ma_forecaster.predict(X_test)
    ma_metrics = compute_prediction_metrics(y_test, ma_preds)

    # CPU specific metrics
    lstm_cpu_mae = float(np.mean(np.abs(lstm_preds[:, 0] - y_test[:, 0])))
    lstm_mem_mae = float(np.mean(np.abs(lstm_preds[:, 1] - y_test[:, 1])))

    logger.info(f"[{horizon_name}] LSTM Overall - MAE: {lstm_metrics['mae']:.5f}, RMSE: {lstm_metrics['rmse']:.5f}, R2: {lstm_metrics['r2']:.4f}")
    logger.info(f"[{horizon_name}] Persistence  - MAE: {pers_metrics['mae']:.5f}, RMSE: {pers_metrics['rmse']:.5f}, R2: {pers_metrics['r2']:.4f}")
    logger.info(f"[{horizon_name}] Moving Avg   - MAE: {ma_metrics['mae']:.5f}, RMSE: {ma_metrics['rmse']:.5f}, R2: {ma_metrics['r2']:.4f}")

    # Save test predictions
    np.savez_compressed(
        f"results/predictions/preds_{horizon_name}.npz",
        y_true=y_test,
        y_pred_lstm=lstm_preds,
        y_pred_pers=pers_preds,
        y_pred_ma=ma_preds
    )

    return {
        "horizon": horizon_name,
        "minutes": horizon_steps * 5,
        "lstm_mae": lstm_metrics["mae"],
        "lstm_rmse": lstm_metrics["rmse"],
        "lstm_mape": lstm_metrics["mape"],
        "lstm_r2": lstm_metrics["r2"],
        "lstm_cpu_mae": round(lstm_cpu_mae, 5),
        "lstm_mem_mae": round(lstm_mem_mae, 5),
        "pers_mae": pers_metrics["mae"],
        "pers_rmse": pers_metrics["rmse"],
        "pers_r2": pers_metrics["r2"],
        "ma_mae": ma_metrics["mae"],
        "ma_rmse": ma_metrics["rmse"],
        "ma_r2": ma_metrics["r2"]
    }


def main():
    set_seed(42)
    cfg = load_yaml_config("configs/lstm.yaml")
    data_path = "data/processed/subset_a_prediction_timeseries.parquet"

    logger.info("Loading time series dataset from %s", data_path)
    df = pd.read_parquet(data_path)

    # Prepare multivariate sequences [cpu_util, mem_util]
    data_matrix = df[["cpu_util", "mem_util"]].values
    scaler = ResourceScaler(feature_range=(0.0, 1.0))
    scaled_data = scaler.fit_transform(data_matrix)

    history_len = 12 # 60 minutes history
    horizons = {
        "10m": 2,
        "30m": 6,
        "60m": 12
    }

    results = []
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    for h_name, h_step in horizons.items():
        X, y = create_sliding_windows(scaled_data, history_len=history_len, horizon=h_step)

        # Chronological split: 70% train, 10% val, 20% test
        n_total = len(X)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.10)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
        X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]

        logger.info(f"Window shapes for {h_name} -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        res = train_and_evaluate_horizon(
            X_train, y_train, X_val, y_val, X_test, y_test,
            horizon_name=h_name, horizon_steps=h_step, cfg=cfg
        )
        results.append(res)

    results_df = pd.DataFrame(results)
    out_table = "results/tables/table5_prediction_performance.csv"
    results_df.to_csv(out_table, index=False)
    logger.info("Prediction performance saved to %s", out_table)
    print("\n" + "="*80)
    print("TABLE 5: WORKLOAD PREDICTION PERFORMANCE COMPARISON")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

