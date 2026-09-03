"""
AutoEncoder for Anomaly Detection.

Train only on normal (benign) data. High reconstruction error = anomaly.

Architecture:
    Encoder:  Input(3073) -> Linear(1536) -> LN -> GELU -> Dropout
                            -> Linear(768)  -> LN -> GELU -> Dropout
                            -> Linear(128)  -> LN -> GELU

    Decoder:  Latent(128)  -> Linear(768)  -> LN -> GELU -> Dropout
                            -> Linear(1536) -> LN -> GELU -> Dropout
                            -> Linear(3073)

    Anomaly Score = MSE(input, reconstruction)
"""

import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
        hidden1 = max(512, input_dim // 2)
        hidden2 = max(256, input_dim // 4)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.LayerNorm(hidden2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden2, hidden1),
            nn.LayerNorm(hidden1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon
