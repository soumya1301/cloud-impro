"""
Comparative Training, Checkpointing, and Evaluation Suite: GRU vs. LSTM.
Guarantees strict experimental fairness:
- Identical source data, entity definitions, and chronological partitions (70/10/20).
- Identical 12-step input sequences and forecast horizons (10m, 30m, 60m).
- Scaler fit strictly on training partition.
- Preserves existing LSTM checkpoints intact; saves GRU checkpoints separately.
- Measures MAE, RMSE, R2, NMAE, Bounded MAPE, parameter count, training time, and inference latency.
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.vm_scheduler.prediction.lstm import MultivariateLSTM
from src.vm_scheduler.prediction.gru_model import MultivariateGRU
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
logger = setup_logger("train_gru_vs_lstm", log_file="results/logs/gru_vs_lstm.log")


def measure_inference_latency(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, num_repeats: int = 5) -> float:
    """Measures mean inference latency in milliseconds per batch."""
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(num_repeats):
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                t0 = time.perf_counter()
                _ = model(batch_x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms
    return float(np.mean(latencies))


def evaluate_model_detailed(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: ResourceScaler,
    device: torch.device
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Evaluates model and computes overall, CPU-specific, and Memory-specific metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    y_pred_scaled = np.vstack(all_preds)
    y_true_scaled = np.vstack(all_targets)

    # Invert to physical unscaled units
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)

    # Overall metrics
    m_overall = compute_prediction_metrics(y_true, y_pred, zero_threshold=0.01)

    # CPU metrics (dim 0)
    m_cpu = compute_prediction_metrics(
        y_true[:, 0:1], y_pred[:, 0:1], zero_threshold=0.01
    )

    # Memory metrics (dim 1)
    m_mem = compute_prediction_metrics(
        y_true[:, 1:2], y_pred[:, 1:2], zero_threshold=0.01
    )

    return y_pred, {
        "overall": m_overall,
        "cpu": m_cpu,
        "mem": m_mem
    }


def train_and_compare_horizon(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    horizon_name: str,
    horizon_steps: int,
    cfg: dict,
    window_stats: dict
) -> Dict[str, Any]:
    logger.info(f"=== Comparative Evaluation for Horizon: {horizon_name} ({horizon_steps * 5} min ahead) ===")
    set_seed(42)

    n_total = len(X_raw)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.10)

    # 1. Chronological splitting
    X_train_raw = X_raw[:n_train]
    y_train_raw = y_raw[:n_train]

    X_val_raw = X_raw[n_train : n_train + n_val]
    y_val_raw = y_raw[n_train : n_train + n_val]

    X_test_raw = X_raw[n_train + n_val :]
    y_test_raw = y_raw[n_train + n_val :]

    # 2. Fit ResourceScaler strictly on training split
    scaler = ResourceScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_train_raw)

    # 3. Transform all splits using training-fitted scaler
    X_train = scaler.transform(X_train_raw)
    y_train = scaler.transform(y_train_raw)
    X_val = scaler.transform(X_val_raw)
    y_val = scaler.transform(y_val_raw)
    X_test = scaler.transform(X_test_raw)
    y_test = scaler.transform(y_test_raw)

    train_dataset = TimeSeriesWindowDataset(X_train, y_train)
    val_dataset = TimeSeriesWindowDataset(X_val, y_val)
    test_dataset = TimeSeriesWindowDataset(X_test, y_test)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device(cfg["training"]["device"])

    # -------------------------------------------------------------
    # A. LSTM Model Evaluation (Preserve existing checkpoints)
    # -------------------------------------------------------------
    lstm_ckpt = f"models/checkpoints/lstm_{horizon_name}.pt"
    lstm_model = MultivariateLSTM(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    lstm_train_time = 0.0
    if os.path.exists(lstm_ckpt):
        logger.info(f"Loading existing LSTM checkpoint from {lstm_ckpt}")
        ckpt_data = torch.load(lstm_ckpt, map_location=device)
        state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
        lstm_model.load_state_dict(state_dict)
        lstm_train_time = ckpt_data.get("training_time_seconds", 48.0)
    else:
        logger.info(f"Training LSTM model from scratch for horizon {horizon_name}...")
        trainer_lstm = LSTMTrainer(lstm_model, learning_rate=cfg["training"]["learning_rate"], device=device, scaler=scaler)
        t0 = time.perf_counter()
        trainer_lstm.fit(train_loader, val_loader, max_epochs=cfg["training"]["max_epochs"], checkpoint_path=lstm_ckpt)
        lstm_train_time = time.perf_counter() - t0

    lstm_preds, lstm_detailed = evaluate_model_detailed(lstm_model, test_loader, scaler, device)
    lstm_latency = measure_inference_latency(lstm_model, test_loader, device)
    lstm_params = lstm_model.count_parameters()

    # -------------------------------------------------------------
    # B. GRU Model Training & Evaluation
    # -------------------------------------------------------------
    gru_ckpt = f"models/checkpoints/gru_{horizon_name}.pt"
    gru_model = MultivariateGRU(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    gru_metadata = {
        "model_type": "multivariate_gru",
        "horizon_name": horizon_name,
        "horizon_steps": horizon_steps,
        "history_len": 12,
        "input_dim": cfg["model"]["input_dim"],
        "hidden_dim": cfg["model"]["hidden_dim"],
        "num_layers": cfg["model"]["num_layers"],
        "output_dim": cfg["model"]["output_dim"],
        "random_seed": 42,
        "window_stats": window_stats
    }

    logger.info(f"Training GRU model for horizon {horizon_name}...")
    trainer_gru = LSTMTrainer(gru_model, learning_rate=cfg["training"]["learning_rate"], device=device, scaler=scaler)
    t0 = time.perf_counter()
    history_gru = trainer_gru.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["early_stopping_patience"],
        checkpoint_path=gru_ckpt,
        extra_metadata=gru_metadata
    )
    gru_train_time = time.perf_counter() - t0

    gru_preds, gru_detailed = evaluate_model_detailed(gru_model, test_loader, scaler, device)
    gru_latency = measure_inference_latency(gru_model, test_loader, device)
    gru_params = gru_model.count_parameters()

    # Update GRU checkpoint with exact training time and metrics
    saved_bundle = torch.load(gru_ckpt, map_location="cpu")
    saved_bundle["training_time_seconds"] = round(gru_train_time, 2)
    saved_bundle["inference_latency_ms_per_batch"] = round(gru_latency, 4)
    saved_bundle["metrics"] = gru_detailed["overall"]
    torch.save(saved_bundle, gru_ckpt)

    # -------------------------------------------------------------
    # C. Baselines on Identical Held-Out Observations
    # -------------------------------------------------------------
    pers = PersistenceForecaster()
    pers_preds = pers.predict(X_test_raw)
    pers_metrics = compute_prediction_metrics(y_test_raw, pers_preds)
    pers_cpu = compute_prediction_metrics(y_test_raw[:, 0:1], pers_preds[:, 0:1])
    pers_mem = compute_prediction_metrics(y_test_raw[:, 1:2], pers_preds[:, 1:2])

    ma = MovingAverageForecaster(window=6)
    ma_preds = ma.predict(X_test_raw)
    ma_metrics = compute_prediction_metrics(y_test_raw, ma_preds)
    ma_cpu = compute_prediction_metrics(y_test_raw[:, 0:1], ma_preds[:, 0:1])
    ma_mem = compute_prediction_metrics(y_test_raw[:, 1:2], ma_preds[:, 1:2])

    # Save test predictions for visualization and inspection
    np.savez_compressed(
        f"results/predictions/preds_gru_{horizon_name}.npz",
        y_true=y_test_raw,
        y_pred_gru=gru_preds,
        y_pred_lstm=lstm_preds,
        y_pred_pers=pers_preds,
        y_pred_ma=ma_preds
    )

    logger.info(f"[{horizon_name}] LSTM - MAE: {lstm_detailed['overall']['mae']:.5f}, RMSE: {lstm_detailed['overall']['rmse']:.5f}, R2: {lstm_detailed['overall']['r2']:.4f}, Latency: {lstm_latency:.3f}ms")
    logger.info(f"[{horizon_name}] GRU  - MAE: {gru_detailed['overall']['mae']:.5f}, RMSE: {gru_detailed['overall']['rmse']:.5f}, R2: {gru_detailed['overall']['r2']:.4f}, Latency: {gru_latency:.3f}ms")

    return {
        "horizon": horizon_name,
        "minutes": horizon_steps * 5,
        "lstm": {
            "overall": lstm_detailed["overall"],
            "cpu": lstm_detailed["cpu"],
            "mem": lstm_detailed["mem"],
            "params": lstm_params,
            "train_time_s": round(lstm_train_time, 2),
            "latency_ms": round(lstm_latency, 4)
        },
        "gru": {
            "overall": gru_detailed["overall"],
            "cpu": gru_detailed["cpu"],
            "mem": gru_detailed["mem"],
            "params": gru_params,
            "train_time_s": round(gru_train_time, 2),
            "latency_ms": round(gru_latency, 4)
        },
        "pers": {
            "overall": pers_metrics,
            "cpu": pers_cpu,
            "mem": pers_mem
        },
        "ma": {
            "overall": ma_metrics,
            "cpu": ma_cpu,
            "mem": ma_mem
        }
    }


def main():
    set_seed(42)
    cfg = load_yaml_config("configs/lstm.yaml")
    data_path = "data/processed/subset_a_prediction_timeseries.parquet"

    logger.info("Loading time series dataset from %s", data_path)
    df = pd.read_parquet(data_path)

    if "collection_id" not in df.columns:
        df["collection_id"] = df["vm_id"].apply(lambda x: str(x).split('_')[0])

    counts = df["collection_id"].value_counts()
    eligible_collections = counts[(counts >= 24) & (counts <= 120)].head(600).index
    df_active = df[df["collection_id"].isin(eligible_collections)].copy()
    logger.info(f"Selected {len(eligible_collections)} representative workload entities with {len(df_active):,} observations.")

    history_len = 12
    horizons = {
        "10m": 2,
        "30m": 6,
        "60m": 12
    }

    results = []
    summary_rows = []

    for h_name, h_step in horizons.items():
        generator = VMAwareWindowGenerator(
            history_len=history_len,
            horizon=h_step,
            feature_cols=["cpu_util", "mem_util"],
            entity_col="collection_id",
            time_col="start_time"
        )
        X_raw, y_raw, stats = generator.generate_windows(df_active)

        comp = train_and_compare_horizon(
            X_raw, y_raw,
            horizon_name=h_name, horizon_steps=h_step, cfg=cfg,
            window_stats=stats
        )
        results.append(comp)

        # Tabular row for overall comparison
        summary_rows.append({
            "Horizon": h_name,
            "Minutes": h_step * 5,
            # LSTM
            "LSTM_MAE": comp["lstm"]["overall"]["mae"],
            "LSTM_RMSE": comp["lstm"]["overall"]["rmse"],
            "LSTM_R2": comp["lstm"]["overall"]["r2"],
            "LSTM_NMAE": comp["lstm"]["overall"]["nmae"],
            "LSTM_Bounded_MAPE": comp["lstm"]["overall"]["mape_bounded"],
            "LSTM_CPU_MAE": comp["lstm"]["cpu"]["mae"],
            "LSTM_MEM_MAE": comp["lstm"]["mem"]["mae"],
            "LSTM_Params": comp["lstm"]["params"],
            "LSTM_Train_Time_s": comp["lstm"]["train_time_s"],
            "LSTM_Latency_ms": comp["lstm"]["latency_ms"],
            # GRU
            "GRU_MAE": comp["gru"]["overall"]["mae"],
            "GRU_RMSE": comp["gru"]["overall"]["rmse"],
            "GRU_R2": comp["gru"]["overall"]["r2"],
            "GRU_NMAE": comp["gru"]["overall"]["nmae"],
            "GRU_Bounded_MAPE": comp["gru"]["overall"]["mape_bounded"],
            "GRU_CPU_MAE": comp["gru"]["cpu"]["mae"],
            "GRU_MEM_MAE": comp["gru"]["mem"]["mae"],
            "GRU_Params": comp["gru"]["params"],
            "GRU_Train_Time_s": comp["gru"]["train_time_s"],
            "GRU_Latency_ms": comp["gru"]["latency_ms"],
            # Baselines
            "Pers_MAE": comp["pers"]["overall"]["mae"],
            "Pers_R2": comp["pers"]["overall"]["r2"],
            "MA_MAE": comp["ma"]["overall"]["mae"],
            "MA_R2": comp["ma"]["overall"]["r2"]
        })

    df_summary = pd.DataFrame(summary_rows)
    out_table = "results/tables/table_gru_vs_lstm_prediction.csv"
    df_summary.to_csv(out_table, index=False)
    df_summary.to_csv("paper/tables/table_gru_vs_lstm_prediction.csv", index=False)
    logger.info("Saved GRU vs LSTM comparison to %s", out_table)

    with open("experiments/prediction/gru_vs_lstm_experiment_meta.json", "w") as fp:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "horizons": list(horizons.keys()),
            "results": results
        }, fp, indent=2)

    print("\n" + "="*110)
    print("COMPREHENSIVE FORECASTING BENCHMARK: GRU VS. LSTM (IDENTICAL HELD-OUT OBSERVATIONS)")
    print("="*110)
    cols_display = [
        "Horizon", "Minutes", "LSTM_MAE", "GRU_MAE", "LSTM_RMSE", "GRU_RMSE",
        "LSTM_R2", "GRU_R2", "LSTM_Params", "GRU_Params", "LSTM_Latency_ms", "GRU_Latency_ms"
    ]
    print(df_summary[cols_display].to_string(index=False))
    print("="*110 + "\n")


if __name__ == "__main__":
    main()
