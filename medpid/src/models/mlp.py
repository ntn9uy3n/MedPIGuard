"""
MLP (Multi-Layer Perceptron) Classifier.

Supervised binary classification: normal (0) vs attack (1).

Architecture:
    Input(3073) -> [Linear -> LayerNorm -> GELU -> Dropout] x N
               -> Linear(1) -> logit
"""

import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
