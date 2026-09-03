"""
Training pipeline for all model types.

Functions:
    train_supervised  — MLP, LSTM-MLP, BiLSTM (binary classification)
    train_autoencoder — AutoEncoder (unsupervised anomaly detection)
    predict_supervised — inference for supervised models
    predict_autoencoder — inference for autoencoder (anomaly scores)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.dataset import FeatureDataset
from src.utils.metrics import evaluate_predictions, find_best_threshold


# ============================================================
# Supervised Training (MLP, LSTM-MLP, BiLSTM)
# ============================================================
def train_supervised(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    patience: int = 10,
    model_name: str = "Model",
) -> nn.Module:
    """
    Train a supervised classifier with early stopping on validation F1.

    Args:
        model: PyTorch model (forward returns logit)
        X_train, y_train: training features and labels
        X_val, y_val: validation features and labels
        epochs: max training epochs
        lr: learning rate
        batch_size: mini-batch size
        weight_decay: AdamW weight decay
        device: 'cuda' or 'cpu'
        patience: early stopping patience
        model_name: name for logging

    Returns:
        Trained model (best checkpoint restored)
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    train_loader = DataLoader(
        FeatureDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )

    best_f1 = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_features = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_logits = model(val_features)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)

        val_f1 = precision_recall_fscore_support(
            y_val, val_preds, average="binary", zero_division=0
        )[2]
        val_acc = accuracy_score(y_val, val_preds)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [{model_name}] Epoch {epoch}/{epochs} | "
                f"Loss: {np.mean(epoch_losses):.6f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch} (best F1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model


# ============================================================
# AutoEncoder Training (Unsupervised)
# ============================================================
def train_autoencoder(
    model: nn.Module,
    X_train_normal: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    Train an AutoEncoder on normal data only (unsupervised).

    Args:
        model: AutoEncoder model
        X_train_normal: normal-only features
        epochs: training epochs
        lr: learning rate
        batch_size: mini-batch size
        weight_decay: AdamW weight decay
        device: 'cuda' or 'cpu'

    Returns:
        Trained model
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    train_loader = DataLoader(
        FeatureDataset(X_train_normal), batch_size=batch_size, shuffle=True
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for features in train_loader:
            features = features.to(device)
            recon = model(features)
            loss = criterion(recon, features)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [AutoEncoder] Epoch {epoch}/{epochs} | Loss: {np.mean(epoch_losses):.6f}")

    model.to(device)
    return model


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def predict_supervised(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Predict probabilities for supervised models.

    Returns:
        [N] array of probabilities (sigmoid output)
    """
    model.eval()
    model.to(device)
    loader = DataLoader(FeatureDataset(X), batch_size=batch_size, shuffle=False)

    all_probs = []
    for features in loader:
        features = features.to(device)
        logits = model(features)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())

    return np.array(all_probs)


@torch.no_grad()
def predict_autoencoder(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute anomaly scores for AutoEncoder.

    Score = MSE(input, reconstruction) per sample.

    Returns:
        [N] array of anomaly scores
    """
    model.eval()
    model.to(device)
    loader = DataLoader(FeatureDataset(X), batch_size=batch_size, shuffle=False)

    scores = []
    for features in loader:
        features = features.to(device)
        recon = model(features)
        error = torch.mean((features - recon) ** 2, dim=1)
        scores.extend(error.cpu().numpy().tolist())

    return np.array(scores)


# ============================================================
# Supervised Training with Weighted Loss & Label Smoothing
# ============================================================
def train_supervised_weighted(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    patience: int = 15,
    model_name: str = "Model",
    pos_weight: float = 3.0,
    label_smoothing: float = 0.05,
    warmup_epochs: int = 3,
) -> nn.Module:
    """
    Train a supervised classifier with:
    - Weighted BCE loss (pos_weight upweights minority class)
    - Label smoothing to reduce overconfidence
    - Warmup + cosine annealing scheduler
    - Threshold optimization on validation for better generalization

    Args:
        model: PyTorch model (forward returns logit)
        X_train, y_train: training features and labels
        X_val, y_val: validation features and labels
        pos_weight: weight for positive (attack) class in BCE
        label_smoothing: smoothing factor for labels
        warmup_epochs: linear warmup epochs before cosine decay

    Returns:
        Trained model (best checkpoint restored)
    """
    model.to(device)

    # Weighted BCE with logits
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )

    # Warmup + Cosine schedule
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )

    # Smooth labels for training
    y_train_smooth = np.where(y_train == 1, 1.0 - label_smoothing, label_smoothing).astype(np.float32)

    train_loader = DataLoader(
        FeatureDataset(X_train, y_train_smooth), batch_size=batch_size, shuffle=True
    )

    best_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation with threshold search
        model.eval()
        with torch.no_grad():
            val_features = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_logits = model(val_features)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()

        # Try multiple thresholds and pick best F1
        best_val_f1 = 0
        best_val_thresh = 0.5
        for thresh in np.arange(0.2, 0.8, 0.02):
            val_preds_t = (val_probs >= thresh).astype(int)
            f1_t = precision_recall_fscore_support(
                y_val, val_preds_t, average="binary", zero_division=0
            )[2]
            if f1_t > best_val_f1:
                best_val_f1 = f1_t
                best_val_thresh = thresh

        val_preds = (val_probs >= best_val_thresh).astype(int)
        val_acc = accuracy_score(y_val, val_preds)

        current_lr = scheduler.get_last_lr()[0]

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  [{model_name}] Epoch {epoch}/{epochs} | "
                f"Loss: {np.mean(epoch_losses):.6f} | LR: {current_lr:.6f} | "
                f"Val Acc: {val_acc:.4f} | Val F1: {best_val_f1:.4f} | Thresh: {best_val_thresh:.2f}"
            )

        if best_val_f1 > best_f1:
            best_f1 = best_val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = best_val_thresh
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{model_name}] Early stopping at epoch {epoch} (best F1={best_f1:.4f}, thresh={best_threshold:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, best_threshold


# ============================================================
# Full evaluation helper
# ============================================================
def evaluate_model(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    model_name: str,
    device: str = "cuda",
    is_autoencoder: bool = False,
    threshold: float = None,
) -> dict:
    """
    Evaluate a model on validation data and return metrics dict.

    For supervised models: use provided threshold or default 0.5.
    For autoencoder: find_best_threshold or use provided threshold.
    """
    if is_autoencoder:
        scores = predict_autoencoder(model, X_val, device=device)
        if threshold is None:
            threshold, _ = find_best_threshold(scores, y_val)
        preds = (scores > threshold).astype(int)
    else:
        probs = predict_supervised(model, X_val, device=device)
        scores = probs
        if threshold is None:
            threshold = 0.5
        preds = (probs >= threshold).astype(int)

    metrics = evaluate_predictions(y_val, preds, scores, method_name=model_name)
    metrics["threshold"] = round(float(threshold), 6) if threshold else None
    return metrics
