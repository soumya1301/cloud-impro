"""
Model trainer for multivariate LSTM with early stopping and learning rate scheduling.
Supports complete checkpoint bundling including model state, scaler parameters, and metadata.
"""
from typing import Dict, Any, Tuple, Optional
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .lstm import MultivariateLSTM
from .metrics import compute_prediction_metrics
from ..preprocessing.normalization import ResourceScaler


class LSTMTrainer:
    """Handles training, validation, early stopping, and checkpointing for MultivariateLSTM."""

    def __init__(
        self,
        model: MultivariateLSTM,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        scaler: Optional[ResourceScaler] = None
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.scaler = scaler
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
        patience: int = 10,
        checkpoint_path: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        best_val_loss = float("inf")
        best_model_weights = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, max_epochs + 1):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * len(batch_x)

            train_loss /= len(train_loader.dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    preds = self.model(batch_x)
                    loss = self.criterion(preds, batch_y)
                    val_loss += loss.item() * len(batch_x)

            val_loss /= len(val_loader.dataset)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0

                if checkpoint_path:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    bundle = {
                        "model_state_dict": best_model_weights,
                        "best_val_loss": float(best_val_loss),
                        "epoch": epoch,
                        "scaler_state": self.scaler.get_state() if self.scaler is not None else None
                    }
                    if extra_metadata:
                        bundle.update(extra_metadata)
                    torch.save(bundle, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        return history

    def evaluate(
        self,
        test_loader: DataLoader,
        scaler: Optional[ResourceScaler] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluates model on test_loader.
        If scaler is provided (or configured on trainer), inverts predictions and targets
        to compute metrics on physical unscaled units.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                preds = self.model(batch_x)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.numpy())

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        active_scaler = scaler or self.scaler
        if active_scaler is not None:
            y_pred_unscaled = active_scaler.inverse_transform(y_pred)
            y_true_unscaled = active_scaler.inverse_transform(y_true)
            metrics = compute_prediction_metrics(y_true_unscaled, y_pred_unscaled)
            return y_pred_unscaled, y_true_unscaled, metrics
        else:
            metrics = compute_prediction_metrics(y_true, y_pred)
            return y_pred, y_true, metrics
