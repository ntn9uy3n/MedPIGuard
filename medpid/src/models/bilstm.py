"""
BiLSTM Classifier (direct on feature vector treated as 1-step sequence).

Alternative to MLP: uses a single-layer BiLSTM on the full feature vector
treated as a 1-step sequence, followed by an MLP head.

Architecture:
    Input(3073) -> Linear(512) -> GELU -> Dropout
               -> reshape [B, 1, 512]
               -> BiLSTM(hidden=256)
               -> MLP head -> logit
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Project input to a lower dimension
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out_dim = lstm_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        projected = self.projection(x)  # [B, proj_dim]
        seq = projected.unsqueeze(1)    # [B, 1, proj_dim]

        lstm_out, (hn, _) = self.lstm(seq)

        combined = hn.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
        fused = torch.cat([combined[-1, 0], combined[-1, 1]], dim=1)

        out = self.head(fused).squeeze(-1)
        return out
