"""
Data loading, splitting, and feature preparation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.features import build_pair_features


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV with columns: doctor_question, patient_answer, [label].

    Returns:
        DataFrame with cleaned text columns.
    """
    df = pd.read_csv(csv_path)

    required = ["doctor_question", "patient_answer"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["doctor_question"] = df["doctor_question"].fillna("").astype(str)
    df["patient_answer"] = df["patient_answer"].fillna("").astype(str)

    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    return df


def split_supervised(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple:
    """
    Stratified train/val split for supervised models.

    Uses ALL data (normal + attack) with stratification.

    Returns:
        (X_train, X_val, y_train, y_val) — indices
    """
    indices = np.arange(len(df))
    labels = df["label"].values
    train_idx, val_idx = train_test_split(
        indices, test_size=(1 - train_ratio), random_state=seed, stratify=labels
    )
    return train_idx, val_idx


def split_autoencoder(
    df: pd.DataFrame,
    train_normal_ratio: float = 0.8,
    seed: int = 42,
) -> tuple:
    """
    Split for AutoEncoder (unsupervised):
      - Train: normal samples only (80%)
      - Eval: remaining normal + all attack

    Returns:
        (train_idx, eval_idx)
    """
    normal_df = df[df["label"] == 0]
    attack_df = df[df["label"] == 1]

    normal_indices = normal_df.index.values
    n_train = int(len(normal_indices) * train_normal_ratio)

    rng = np.random.RandomState(seed)
    rng.shuffle(normal_indices)

    train_idx = normal_indices[:n_train]
    eval_idx = np.concatenate([normal_indices[n_train:], attack_df.index.values])
    rng.shuffle(eval_idx)

    return train_idx, eval_idx


def prepare_features(
    df_subset,
    hq: np.ndarray,
    ha: np.ndarray,
    scaler: StandardScaler = None,
    fit: bool = False,
) -> tuple:
    """
    Build fused features from embeddings, optionally fit/apply scaler.

    Args:
        df_subset: DataFrame subset (to select rows from hq/ha)
        hq: full [N, D] doctor embeddings
        ha: full [N, D] patient embeddings
        scaler: fitted StandardScaler (if fit=False)
        fit: if True, fit scaler on this data

    Returns:
        (X_fused_scaled, scaler)
    """
    idx = df_subset.index.values
    hq_sub = hq[idx]
    ha_sub = ha[idx]

    z = build_pair_features(hq_sub, ha_sub)

    if fit:
        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z).astype(np.float32)
    else:
        z_scaled = scaler.transform(z).astype(np.float32)

    return z_scaled, scaler
