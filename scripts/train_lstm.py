"""
Multivariate LSTM training and evaluation script across 10m, 30m, and 60m forecast horizons.
Enforces strict entity isolation (no cross-VM window leakage) and leakage-safe scaling
(scaler fit strictly on the training partition, with parameters saved in the checkpoint).
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.vm_scheduler.prediction.lstm import MultivariateLSTM
from src.vm_scheduler.prediction.trainer import LSTMTrainer
from src.vm_scheduler.prediction.baselines import PersistenceForecaster, MovingAverageForecaster
from src.vm_scheduler.prediction.metrics import compute_prediction_metrics
from src.vm_scheduler.preprocessing.windowing import VMAwareWindowGenerator, TimeSeriesWindowDataset
from src.vm_scheduler.preprocessing.normalization import ResourceScaler
from src.vm_scheduler.utils.config import setup_logger, set_seed, load_yaml_config

os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/predictions", exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("experiments/prediction", exist_ok=True)
os.makedirs("paper/tables", exist_ok=True)
logger = setup_logger("train_lstm", log_file="results/logs/train_lstm.log")


def train_and_evaluate_horizon(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    horizon_name: str,
    horizon_steps: int,
    cfg: dict,
    window_stats: dict
):
    logger.info(f"--- Training LSTM for Horizon: {horizon_name} ({horizon_steps * 5} min ahead) ---")

    n_total = len(X_raw)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)

    # 1. Chronological partition BEFORE fitting scaler (strict anti-leakage guarantee)
    X_train_raw = X_raw[:n_train]
    y_train_raw = y_raw[:n_train]

    X_val_raw = X_raw[n_train : n_train + n_val]
    y_val_raw = y_raw[n_train : n_train + n_val]

    X_test_raw = X_raw[n_train + n_val :]
    y_test_raw = y_raw[n_train + n_val :]

    # 2. Fit ResourceScaler strictly on the training partition
    scaler = ResourceScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_train_raw)

    # 3. Transform all splits using the training-fitted scaler
    X_train = scaler.transform(X_train_raw)
    y_train = scaler.transform(y_train_raw)

    X_val = scaler.transform(X_val_raw)
    y_val = scaler.transform(y_val_raw)

    X_test = scaler.transform(X_test_raw)
    y_test = scaler.transform(y_test_raw)

    logger.info(f"Split sizes ({horizon_name}) -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

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
        device=cfg["training"]["device"],
        scaler=scaler
    )

    metadata = {
        "horizon_name": horizon_name,
        "horizon_steps": horizon_steps,
        "history_len": 12,
        "input_dim": cfg["model"]["input_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "num_layers": cfg["model"]["num_layers"],
        "output_dim": cfg["model"]["output_dim"],
        "window_stats": window_stats
    }

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["early_stopping_patience"],
        checkpoint_path=ckpt_path,
        extra_metadata=metadata
    )

    # 4. Evaluate LSTM on held-out test set in physical unscaled units
    lstm_preds_unscaled, true_vals_unscaled, lstm_metrics = trainer.evaluate(test_loader, scaler=scaler)

    # 5. Evaluate Baselines on the exact same test set (unscaled physical units)
    pers_forecaster = PersistenceForecaster()
    pers_preds_unscaled = pers_forecaster.predict(X_test_raw)
    pers_metrics = compute_prediction_metrics(y_test_raw, pers_preds_unscaled)

    ma_forecaster = MovingAverageForecaster(window=6)
    ma_preds_unscaled = ma_forecaster.predict(X_test_raw)
    ma_metrics = compute_prediction_metrics(y_test_raw, ma_preds_unscaled)

    # Dimension-specific MAE
    lstm_cpu_mae = float(np.mean(np.abs(lstm_preds_unscaled[:, 0] - y_test_raw[:, 0])))
    lstm_mem_mae = float(np.mean(np.abs(lstm_preds_unscaled[:, 1] - y_test_raw[:, 1])))

    logger.info(f"[{horizon_name}] LSTM Overall - MAE: {lstm_metrics['mae']:.5f}, RMSE: {lstm_metrics['rmse']:.5f}, R2: {lstm_metrics['r2']:.4f}, NMAE: {lstm_metrics['nmae']:.4f}, Bounded MAPE: {lstm_metrics['mape_bounded']:.1f}%")
    logger.info(f"[{horizon_name}] Persistence  - MAE: {pers_metrics['mae']:.5f}, RMSE: {pers_metrics['rmse']:.5f}, R2: {pers_metrics['r2']:.4f}, Bounded MAPE: {pers_metrics['mape_bounded']:.1f}%")
    logger.info(f"[{horizon_name}] Moving Avg   - MAE: {ma_metrics['mae']:.5f}, RMSE: {ma_metrics['rmse']:.5f}, R2: {ma_metrics['r2']:.4f}, Bounded MAPE: {ma_metrics['mape_bounded']:.1f}%")

    # Save test predictions
    np.savez_compressed(
        f"results/predictions/preds_{horizon_name}.npz",
        y_true=y_test_raw,
        y_pred_lstm=lstm_preds_unscaled,
        y_pred_pers=pers_preds_unscaled,
        y_pred_ma=ma_preds_unscaled
    )

    return {
        "horizon": horizon_name,
        "minutes": horizon_steps * 5,
        "lstm_mae": lstm_metrics["mae"],
        "lstm_rmse": lstm_metrics["rmse"],
        "lstm_nmae": lstm_metrics["nmae"],
        "lstm_mape_raw": lstm_metrics["mape"],
        "lstm_mape_bounded": lstm_metrics["mape_bounded"],
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

    # Derive entity column: collection_id represents the Borg workload/allocation entity
    if "collection_id" not in df.columns:
        df["collection_id"] = df["vm_id"].apply(lambda x: str(x).split('_')[0])

    # Sample representative active workload entities to train efficiently without cross-entity leakage
    counts = df["collection_id"].value_counts()
    eligible_collections = counts[(counts >= 24) & (counts <= 120)].head(600).index
    df_active = df[df["collection_id"].isin(eligible_collections)].copy()
    logger.info(f"Selected {len(eligible_collections)} representative workload entities with {len(df_active):,} observations.")

    history_len = 12  # 60 minutes history (12 x 5m steps)
    horizons = {
        "10m": 2,
        "30m": 6,
        "60m": 12
    }

    results = []

    for h_name, h_step in horizons.items():
        # Generate windows strictly partitioned by entity (never crossing entity boundaries)
        generator = VMAwareWindowGenerator(
            history_len=history_len,
            horizon=h_step,
            feature_cols=["cpu_util", "mem_util"],
            entity_col="collection_id",
            time_col="start_time"
        )
        X_raw, y_raw, stats = generator.generate_windows(df_active)
        logger.info(f"Window generation stats for {h_name}: {stats}")

        res = train_and_evaluate_horizon(
            X_raw, y_raw,
            horizon_name=h_name, horizon_steps=h_step, cfg=cfg,
            window_stats=stats
        )
        results.append(res)

    results_df = pd.DataFrame(results)
    out_table = "results/tables/table5_prediction_performance.csv"
    results_df.to_csv(out_table, index=False)
    results_df.to_csv("paper/tables/table5_prediction_performance.csv", index=False)
    logger.info("Prediction performance saved to %s", out_table)

    with open("experiments/prediction/prediction_run_meta.json", "w") as fp:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "horizons": list(horizons.keys()),
            "results": results_df.to_dict(orient="records")
        }, fp, indent=2)

    print("\n" + "="*80)
    print("TABLE 5: WORKLOAD PREDICTION PERFORMANCE COMPARISON (LEAKAGE-FREE)")
    print("="*80)
    print(results_df[["horizon", "minutes", "lstm_mae", "lstm_rmse", "lstm_r2", "lstm_mape_bounded", "pers_mae", "ma_mae"]].to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
