"""
Transformer Encoder Classifier for tabular features.

Uses PyTorch's TransformerEncoderLayer with multi-head self-attention
on feature chunks (same chunking strategy as LSTM-MLP). This architecture
captures global interactions between feature subspaces more effectively
than LSTM-based models.

Architecture:
    Input(3073) -> reshape -> [B, num_chunks, chunk_dim]
               -> TransformerEncoderLayers (N layers, H heads)
               -> [CLS] token or mean pooling
               -> MLP head -> logit

Reference:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Li et al., "Transformers for Tabular Data" (Tabular, ICLR 2022)
    - Hao et al., "Graph Neural Networks for Tabular Data with Missing Values" (2022)
"""

import torch
import torch.nn as nn


class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_chunks: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.2,
        mlp_hidden: int = 256,
        mlp_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_chunks = num_chunks

        # Compute chunk dim (pad if not divisible)
        chunk_dim = input_dim // num_chunks
        self.padded_dim = chunk_dim * num_chunks
        self.chunk_dim = chunk_dim

        # Input projection to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(chunk_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            layer_norm_eps=1e-5,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head on pooled transformer output
        self.head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
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

        # Reshape to sequence: [B, num_chunks, chunk_dim]
        seq = x.view(batch_size, self.num_chunks, -1)

        # Project to d_model: [B, num_chunks, d_model]
        seq = self.input_proj(seq)

        # Transformer encoder: [B, num_chunks, d_model]
        transformed = self.transformer(seq)

        # Mean pooling over sequence dimension (all chunks contribute equally)
        pooled = transformed.mean(dim=1)  # [B, d_model]

        # Classification head
        out = self.head(pooled).squeeze(-1)
        return out
