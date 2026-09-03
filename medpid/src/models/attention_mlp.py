"""
Attention-MLP Classifier — Optimized for Medical Prompt Injection Detection.

Combines multi-head self-attention over feature chunks with an MLP head.
Designed to be lighter and more generalizable than pure Transformer/TabNet
while capturing inter-feature interactions better than vanilla MLP.

Key design choices for generalization:
1. **Feature chunk attention** — treat 3073-d vector as 4 chunks of 768-d
   (matching the BERT embedding structure) and apply self-attention
2. **Stochastic depth (drop path)** — randomly zero out layer outputs during training
3. **Label smoothing** — softens targets to reduce overconfidence
4. **Smaller capacity** — fewer params than TabNet to reduce overfitting

Architecture:
    Input(3073) -> reshape [B, 4, 768]
               -> MultiHeadAttention (4 heads, 1 layer)
               -> Flatten + Projection
               -> MLP head with dropout -> logit

Reference:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Hu et al., "Transformer in Tabular View: A Unified Architecture" (2023)
    - Stochastic Depth: Huang et al., "Deep Networks with Stochastic Depth" (ICCV 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    """Drop entire layer output during training (Stochastic Depth / DropPath)."""

    def __init__(self, p: float, mode: str = "row"):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class AttentionMLPClassifier(nn.Module):
    """
    Attention-MLP hybrid classifier.

    Splits the fused feature vector into chunks (matching BERT embedding
    subspaces), applies self-attention to learn inter-chunk relationships,
    then passes through a compact MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        num_chunks: int = 4,
        d_model: int = 768,
        nhead: int = 8,
        num_attn_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        stochastic_depth_rate: float = 0.1,
        mlp_hidden: list = None,
        mlp_dropout: float = 0.3,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256]

        self.num_chunks = num_chunks

        # Pad if needed (3073 = 4*768 + 1, drop the cosine scalar for attention)
        chunk_dim = input_dim // num_chunks
        self.padded_dim = chunk_dim * num_chunks

        # Store the extra dim (cosine similarity) separately
        self.extra_dim = input_dim - self.padded_dim

        # Input projection for chunk features
        self.input_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5),
        )

        # Transformer encoder layer(s) for inter-chunk attention
        encoder_layers = []
        for _ in range(num_attn_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    layer_norm_eps=1e-5,
                )
            )
        self.attention_encoder = nn.Sequential(*encoder_layers)

        # Stochastic depth after attention
        self.stochastic_depth = StochasticDepth(stochastic_depth_rate)

        # MLP classification head
        head_layers = []
        prev_dim = d_model * num_chunks + self.extra_dim  # attention output + cosine
        for h_dim in mlp_hidden:
            head_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
            ])
            prev_dim = h_dim
        head_layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        batch_size = x.shape[0]

        # Split into chunks and extra (cosine similarity scalar)
        if self.extra_dim > 0:
            extra = x[:, -self.extra_dim:]  # [B, extra_dim]
            main = x[:, : self.padded_dim]  # [B, padded_dim]
        else:
            extra = None
            main = x

        # Reshape to sequence: [B, num_chunks, chunk_dim]
        chunks = main.view(batch_size, self.num_chunks, -1)

        # Project and apply attention
        chunks = self.input_proj(chunks)
        attended = self.attention_encoder(chunks)  # [B, num_chunks, d_model]
        attended = self.stochastic_depth(attended)

        # Flatten attention output
        flattened = attended.view(batch_size, -1)  # [B, d_model * num_chunks]

        # Concatenate with extra features (cosine similarity)
        if extra is not None:
            fused = torch.cat([flattened, extra], dim=-1)
        else:
            fused = flattened

        # Classification head
        out = self.head(fused).squeeze(-1)
        return out
