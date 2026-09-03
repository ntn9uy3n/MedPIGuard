"""
TabNet-inspired Classifier for tabular features.

Inspired by TabNet (Arlot et al., AAAI 2021) — an attention-based
architecture for interpretable tabular deep learning. Key innovations:

1. **Feature Selection Masks:** Learnable attention over input features
   at each decision step, enabling sparse feature usage.
2. **Sequential Decision Steps:** Multiple gradient steps that
   iteratively refine the representation.
3. **Feature Aggregation:** Pool information from all decision steps.

This is a simplified, classification-oriented version adapted for
the 3073-d fused BiomedBERT feature vectors.

Architecture:
    Input(3073) -> BN
    For each step (n_steps):
        Feature Attention (from prev_step_out -> input_dim logits)
        -> Sparsemax mask -> masked input
        -> Feature Transformer (Linear + PBReLU + BN + Dropout)
        -> Aggregate step output
    -> Classification Head (aggregated steps) -> logit

Reference:
    - Arlot et al., "TabNet: Attentive Interpretable Tabular Learning" (AAAI 2021)
    - Hao et al., "Improving Tabular Data Modeling with Transformer-based Methods" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PBReLu(nn.Module):
    """
    Parameterized ReLU — learns separate slopes for positive and negative inputs.
    f(x) = alpha * max(x, 0) + beta * min(x, 0)
    """

    def __init__(self, features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.ones(features))

    def forward(self, x):
        return F.relu(self.alpha * x) + F.relu(-self.beta * x)


class TabNetFeatureTransformer(nn.Module):
    """Feature transformer block with PBReLU activation."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.01)
        self.activation = PBReLu(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TabNetClassifier(nn.Module):
    """
    TabNet-inspired classifier for prompt injection detection.

    Uses learnable feature selection masks over the 3073-d input,
    followed by sequential decision steps and a classification head.
    """

    def __init__(
        self,
        input_dim: int,
        n_steps: int = 4,
        dims: list = None,
        sparse_sparsity: float = 0.01,
        bn_momentum: float = 0.01,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dims is None:
            dims = [2048, 1024, 512]

        self.n_steps = n_steps
        self.input_dim = input_dim
        self.sparse_sparsity = sparse_sparsity
        self.step_output_dim = dims[-1]

        # Prior for feature selection (uniform initially)
        self.prior = nn.Parameter(torch.zeros(input_dim))

        # Initial BN for input
        self.initial_bn = nn.BatchNorm1d(input_dim, eps=1e-5, momentum=bn_momentum)

        # Feature attention transformers: map step output -> input_dim logits
        # Step 0: input_dim -> input_dim (from initial BN output)
        # Step 1+: step_output_dim -> input_dim (from prev step's feature transformer)
        self.feature_attention_transformers = nn.ModuleList()
        self.feature_attention_transformers.append(
            nn.Linear(input_dim, input_dim, bias=False)
        )
        for _ in range(n_steps - 1):
            self.feature_attention_transformers.append(
                nn.Linear(self.step_output_dim, input_dim, bias=False)
            )

        # Feature transformer for each step (input_dim -> dims[0] -> ... -> dims[-1])
        self.feature_transformers = nn.ModuleList()
        for _ in range(n_steps):
            ft_layers = []
            prev_dim = input_dim
            for d in dims:
                ft_layers.append(nn.Linear(prev_dim, d, bias=False))
                ft_layers.append(nn.BatchNorm1d(d, eps=1e-5, momentum=bn_momentum))
                ft_layers.append(PBReLu(d))
                ft_layers.append(nn.Dropout(dropout))
                prev_dim = d
            self.feature_transformers.append(nn.Sequential(*ft_layers))

        # Classification head on aggregated step outputs
        head_input = self.step_output_dim * n_steps
        self.classifier = nn.Sequential(
            nn.Linear(head_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _sparsemax(self, logits: torch.Tensor) -> torch.Tensor:
        """Sparsemax normalization — produces sparse probabilities."""
        z_sorted, _ = torch.sort(logits, descending=True, dim=-1)
        z_cumsum = torch.cumsum(z_sorted, dim=-1) - 1.0

        k = torch.arange(1, logits.shape[-1] + 1, device=logits.device).unsqueeze(0)
        k_mask = k * z_sorted > z_cumsum

        k_sum = k_mask.int().sum(dim=-1, keepdim=True)

        z_threshold = (
            z_cumsum.gather(-1, (k_sum - 1).clamp(min=0).long()) / k_sum.float()
        ).clamp(min=0)

        mask = F.relu(logits - z_threshold)
        return mask

    def forward(self, x):
        batch_size = x.shape[0]

        # Initial batch norm
        x_bn = self.initial_bn(x)  # [B, input_dim]

        # Collect step outputs for aggregation
        step_outputs = []
        prev_step_out = x_bn  # For step 0, use BN output

        for step in range(self.n_steps):
            # Compute feature selection logits
            if step == 0:
                attention_logits = self.feature_attention_transformers[0](prev_step_out) + self.prior
            else:
                attention_logits = self.feature_attention_transformers[step](prev_step_out) + self.prior

            # Sparsemax mask
            mask = self._sparsemax(attention_logits.tanh())  # [B, input_dim]
            mask = mask * (1 - self.sparse_sparsity)

            # Apply mask to original input
            x_masked = mask * x  # [B, input_dim]

            # Feature transformer
            step_out = self.feature_transformers[step](x_masked)  # [B, step_output_dim]
            step_outputs.append(step_out)

            # Use this step's output for next step's attention
            prev_step_out = step_out

        # Concatenate all step outputs
        aggregated = torch.cat(step_outputs, dim=-1)  # [B, step_output_dim * n_steps]

        # Classification
        out = self.classifier(aggregated).squeeze(-1)
        return out
