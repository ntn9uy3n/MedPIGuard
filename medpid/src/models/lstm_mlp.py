"""
LSTM-MLP Hybrid Classifier.

Treats the fused feature vector as a sequence of chunks, processes
with a BiLSTM, then classifies with an MLP head.

Architecture:
    Input(3073) -> reshape -> [B, 4, chunk_dim]
               -> BiLSTM(hidden=256, layers=2)
               -> concat last hidden (512)
               -> MLP -> logit
"""

import torch
import torch.nn as nn


class LSTMMLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_chunks: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_chunks = num_chunks

        chunk_dim = input_dim // num_chunks
        self.padded_dim = chunk_dim * num_chunks
        self.chunk_dim = chunk_dim

        self.lstm = nn.LSTM(
            input_size=chunk_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional
        self.mlp = nn.Sequential(
            nn.Linear(lstm_out_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # Pad or truncate to padded_dim
        if x.shape[1] < self.padded_dim:
            pad = torch.zeros(batch_size, self.padded_dim - x.shape[1], device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > self.padded_dim:
            x = x[:, : self.padded_dim]

        seq = x.view(batch_size, self.num_chunks, -1)

        lstm_out, (hn, _) = self.lstm(seq)

        # Concat forward + backward hidden from last layer
        combined = hn.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
        last_forward = combined[-1, 0]
        last_backward = combined[-1, 1]
        fused = torch.cat([last_forward, last_backward], dim=1)

        out = self.mlp(fused).squeeze(-1)
        return out
