"""
Inference engine for trained multivariate LSTM workload forecasting models.
Enforces strict model provenance: genuine empirical inference on observed history,
and transparent fallback categorization for insufficient or absent history.
"""
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from .lstm import MultivariateLSTM
from ..preprocessing.normalization import ResourceScaler


class LSTMInferencer:
    """
    Loads a trained PyTorch checkpoint and provides empirical demand predictions
    for VM instances based on their historical observation windows.
    """

    def __init__(
        self,
        checkpoint_path: str,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 2,
        scaler: Optional[ResourceScaler] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.scaler = scaler

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"LSTM checkpoint not found at: {checkpoint_path}")

        loaded = torch.load(ckpt, map_location=self.device)

        # Handle both bundle dictionaries and raw state_dicts
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
            # Restore embedded scaler if not explicitly provided
            if self.scaler is None and "scaler_state" in loaded and loaded["scaler_state"] is not None:
                self.scaler = ResourceScaler.from_state(loaded["scaler_state"])
            # Override architecture params if present in metadata
            input_dim = loaded.get("input_dim", input_dim)
            hidden_dim = loaded.get("hidden_dim", hidden_dim)
            num_layers = loaded.get("num_layers", num_layers)
            output_dim = loaded.get("output_dim", output_dim)
            self.metadata = loaded
        else:
            state_dict = loaded
            self.metadata = {}

        self.model = MultivariateLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, sequence: np.ndarray, scale_input: bool = True) -> Tuple[float, float]:
        """
        Predicts future [CPU, RAM] demand from a single history window of shape (seq_len, 2).
        Returns unscaled physical demand (predicted_cpu, predicted_mem).
        """
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)

        # Scale input sequence if scaler is available and requested
        if scale_input and self.scaler is not None:
            seq_scaled = self.scaler.transform(sequence)
        else:
            seq_scaled = sequence.astype(np.float32)

        tensor_in = torch.tensor(seq_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(tensor_in).cpu().numpy()[0]

        # Invert scaling to physical units
        if self.scaler is not None:
            preds_unscaled = self.scaler.inverse_transform(np.expand_dims(preds_scaled, axis=0))[0]
        else:
            preds_unscaled = preds_scaled

        pred_cpu = float(np.clip(preds_unscaled[0], 0.001, 1.0))
        pred_mem = float(np.clip(preds_unscaled[1], 0.001, 1.0))
        return pred_cpu, pred_mem

    def predict_batch(self, sequences: np.ndarray, scale_input: bool = True) -> np.ndarray:
        """
        Predicts future demands for a batch of sequences of shape (N, seq_len, 2).
        Returns array of unscaled physical demands of shape (N, 2).
        """
        if scale_input and self.scaler is not None:
            seq_scaled = self.scaler.transform(sequences)
        else:
            seq_scaled = sequences.astype(np.float32)

        tensor_in = torch.tensor(seq_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds_scaled = self.model(tensor_in).cpu().numpy()

        if self.scaler is not None:
            preds_unscaled = self.scaler.inverse_transform(preds_scaled)
        else:
            preds_unscaled = preds_scaled

        return np.clip(preds_unscaled, 0.001, 1.0)

    def generate_vm_predictions(
        self,
        vms_df: pd.DataFrame,
        timeseries_df: pd.DataFrame,
        history_len: int = 12,
        entity_col: str = "vm_id"
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
        """
        Maps each VM in `vms_df` to its predicted demand from `timeseries_df`.
        Categorizes VM prediction provenance transparently:
        - Category A: Empirical LSTM predictions (VMs with genuine history >= history_len).
        - Category B: Insufficient-history VMs (history exists but < history_len).
        - Category C: Non-LSTM fallback estimates (requested demand modulated by workload profile,
                      strictly without fabricating artificial synthetic time series).
        """
        predictions: Dict[str, Tuple[float, float]] = {}
        provenance: Dict[str, str] = {}
        counts = {"Category_A_Empirical": 0, "Category_B_Insufficient": 0, "Category_C_Request_Fallback": 0}

        grouped = timeseries_df.groupby(entity_col)

        for idx, row in vms_df.iterrows():
            vm_id = str(row["vm_id"])
            base_cpu = float(row.get("cpu_req", 0.25))
            base_mem = float(row.get("mem_req", 0.10))

            if vm_id in grouped.groups:
                g = grouped.get_group(vm_id)
                if len(g) >= history_len:
                    # Category A: Genuine empirical LSTM forecast
                    sorted_ts = g.sort_values("start_time")
                    seq = sorted_ts[["cpu_util", "mem_util"]].tail(history_len).values
                    pred_cpu, pred_mem = self.predict_single(seq)
                    predictions[vm_id] = (pred_cpu, pred_mem)
                    provenance[vm_id] = "Category_A_Empirical"
                    counts["Category_A_Empirical"] += 1
                else:
                    # Category B: Insufficient history -> mean of observed history
                    obs_cpu = float(g["cpu_util"].mean())
                    obs_mem = float(g["mem_util"].mean())
                    predictions[vm_id] = (max(0.001, obs_cpu), max(0.001, obs_mem))
                    provenance[vm_id] = "Category_B_Insufficient"
                    counts["Category_B_Insufficient"] += 1
            else:
                # Category C: No history in trace -> demand estimate based on requested capacity
                # and profile class without fabricating synthetic time-series windows
                profile_class = int(row.get("profile_class", 0))
                # CPU-intensive class typically operates closer to requested ceiling
                cpu_factor = 0.85 if profile_class in {1, 2} else 0.60
                mem_factor = 0.70
                pred_cpu = float(np.clip(base_cpu * cpu_factor, 0.001, 1.0))
                pred_mem = float(np.clip(base_mem * mem_factor, 0.001, 1.0))
                predictions[vm_id] = (pred_cpu, pred_mem)
                provenance[vm_id] = "Category_C_Request_Fallback"
                counts["Category_C_Request_Fallback"] += 1

        total = len(vms_df)
        provenance_summary = {
            "total_vms": total,
            "category_counts": counts,
            "category_percentages": {k: round(v / total * 100.0, 2) for k, v in counts.items()},
            "provenance_map": provenance
        }

        return predictions, provenance_summary
