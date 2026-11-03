"""
Multivariate Gated Recurrent Unit (GRU) network for VM resource demand forecasting.
Maintains identical input and output conventions with MultivariateLSTM.
"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class MultivariateGRU(nn.Module):
    """
    Multivariate GRU network predicting future multidimensional resource vectors.
    Input shape: (batch_size, seq_len, input_dim)
    Output shape: (batch_size, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.Sigmoid()  # Keeps normalized resource outputs bounded within [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h_n = self.gru(x)
        # Take the hidden state of the final time step
        last_hidden = out[:, -1, :]
        preds = self.fc(last_hidden)
        return preds

    def count_parameters(self) -> int:
        """Returns total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Returns model architectural configuration."""
        return {
            "model_type": "multivariate_gru",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "trainable_parameters": self.count_parameters()
        }

