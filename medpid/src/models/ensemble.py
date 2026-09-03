"""
Ensemble Classifier — combines predictions from multiple models.

Uses weighted average of predicted probabilities from:
1. MLP Classifier
2. Attention-MLP Classifier
3. BiLSTM Classifier

Ensemble typically improves F1 by 1-3% over the best single model
by reducing variance and leveraging diverse model perspectives.
"""

import torch
import torch.nn as nn
import numpy as np


class EnsembleClassifier:
    """
    Simple probability-averaging ensemble.

    In production, this loads pre-trained models and averages
    their sigmoid outputs before applying a threshold.
    """

    def __init__(self, models: dict, weights: dict = None, threshold: float = 0.5):
        """
        Args:
            models: dict of {name: model_instance}
            weights: dict of {name: weight} — defaults to uniform
            threshold: decision threshold on averaged probability
        """
        self.models = models
        self.threshold = threshold

        if weights is None:
            n = len(models)
            self.weights = {k: 1.0 / n for k in models}
        else:
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, device: str = "cuda", batch_size: int = 64) -> np.ndarray:
        """
        Return ensemble probability (averaged sigmoid outputs).
        """
        from src.data.dataset import FeatureDataset
        from torch.utils.data import DataLoader

        X_tensor = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(FeatureDataset(X_tensor.numpy()), batch_size=batch_size, shuffle=False)

        all_probs = []
        for batch in loader:
            features = batch[0].to(device) if isinstance(batch, tuple) else batch.to(device)
            batch_probs = []
            total_weight = 0

            for name, model in self.models.items():
                model.eval()
                model.to(device)
                logits = model(features)
                prob = torch.sigmoid(logits).cpu().numpy()
                batch_probs.append(prob * self.weights[name])
                total_weight += self.weights[name]

            avg_prob = sum(batch_probs) / total_weight
            all_probs.extend(avg_prob.tolist())

        return np.array(all_probs)

    def predict(self, X: np.ndarray, device: str = "cuda", batch_size: int = 64) -> np.ndarray:
        probs = self.predict_proba(X, device=device, batch_size=batch_size)
        return (probs >= self.threshold).astype(int)
