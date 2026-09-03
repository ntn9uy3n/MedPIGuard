"""
Data module: loading, encoding, feature fusion, dataset/dataloader creation.
"""

from src.data.encoder import TextPairEmbedder
from src.data.loader import load_csv, split_supervised, split_autoencoder, prepare_features
from src.data.dataset import FeatureDataset

__all__ = [
    "TextPairEmbedder",
    "load_csv",
    "split_supervised",
    "split_autoencoder",
    "prepare_features",
    "FeatureDataset",
]
