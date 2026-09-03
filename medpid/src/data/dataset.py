"""
PyTorch Dataset for feature vectors.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class FeatureDataset(Dataset):
    """
    Simple dataset wrapping numpy feature arrays.

    Args:
        features: [N, D] numpy array
        labels: [N] numpy array (optional, None for autoencoder)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]
