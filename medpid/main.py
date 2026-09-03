#!/usr/bin/env python3
"""
MedPIDefense — Deep Learning Baseline Runner
=============================================

Train models on trainset.csv and evaluate on both validation and test sets.

Usage:
    # Train + evaluate all models (val + test)
    PYTHONPATH=. python main.py

    # Evaluate trained models on test set only
    PYTHONPATH=. python main.py --test-only

    # Train single model
    PYTHONPATH=. python main.py --model mlp
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from src.config import *
from src.utils import set_seed, ensure_dir, get_device
from src.data.loader import load_csv, split_supervised, split_autoencoder, prepare_features
from src.data.encoder import TextPairEmbedder
from src.models.mlp import MLPClassifier
from src.models.lstm_mlp import LSTMMLPClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.autoencoder import AutoEncoder
from src.models.transformer_encoder import TransformerEncoderClassifier
from src.models.tabnet import TabNetClassifier
from src.models.attention_mlp import AttentionMLPClassifier
from src.train import (
    train_supervised,
    train_supervised_weighted,
    train_autoencoder,
    predict_supervised,
    predict_autoencoder,
    evaluate_model,
)
from src.utils.metrics import evaluate_predictions, find_best_threshold


# ============================================================
# Helpers
# ============================================================
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_trained_model(model_dir: Path, device: str):
    """Load a saved model checkpoint and return (model, config)."""
    ckpt_files = {
        "mlp": model_dir / "mlp" / "mlp_classifier.pt",
        "lstm_mlp": model_dir / "lstm_mlp" / "lstm_mlp_classifier.pt",
        "bilstm": model_dir / "bilstm" / "bilstm_classifier.pt",
        "autoencoder": model_dir / "autoencoder" / "autoencoder.pt",
        "transformer_encoder": model_dir / "transformer_encoder" / "transformer_encoder_classifier.pt",
        "tabnet": model_dir / "tabnet" / "tabnet_classifier.pt",
        "attention_mlp": model_dir / "attention_mlp" / "attention_mlp_classifier.pt",
    }
    return ckpt_files


# ============================================================
# Training Functions
# ============================================================
def run_mlp(input_dim, X_train, y_train, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 1: MLP CLASSIFIER (Supervised, Optimized)")
    print("=" * 70)
    start = time.time()

    model = MLPClassifier(input_dim=input_dim, hidden_dims=MLP_HIDDEN_DIMS, dropout=MLP_DROPOUT)
    print(f"  Parameters: {count_params(model):,}")

    model, best_thresh = train_supervised_weighted(
        model, X_train, y_train, X_val, y_val,
        epochs=MLP_EPOCHS, lr=MLP_LR, batch_size=MLP_BATCH_SIZE,
        weight_decay=MLP_WEIGHT_DECAY, device=device, patience=MLP_PATIENCE,
        model_name="MLP",
        pos_weight=2.5,
        label_smoothing=0.05,
        warmup_epochs=3,
    )

    val_metrics = evaluate_model(model, X_val, y_val, model_name="MLP", device=device, threshold=best_thresh)
    elapsed = time.time() - start

    # Save
    save_dir = output_dir / "mlp"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "hidden_dims": MLP_HIDDEN_DIMS, "dropout": MLP_DROPOUT,
        "method": "mlp_classifier",
    }, save_dir / "mlp_classifier.pt")
    print(f"  Saved to: {save_dir}")

    return model, val_metrics, elapsed


def run_lstm_mlp(input_dim, X_train, y_train, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 2: LSTM-MLP CLASSIFIER (Supervised)")
    print("=" * 70)
    start = time.time()

    model = LSTMMLPClassifier(
        input_dim=input_dim, num_chunks=LSTM_NUM_CHUNKS, lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS, lstm_dropout=LSTM_DROPOUT,
        mlp_hidden=LSTM_MLP_HIDDEN, mlp_dropout=LSTM_MLP_DROPOUT,
    )
    print(f"  Parameters: {count_params(model):,}")

    model = train_supervised(
        model, X_train, y_train, X_val, y_val,
        epochs=LSTM_EPOCHS, lr=LSTM_LR, batch_size=LSTM_BATCH_SIZE,
        weight_decay=LSTM_WEIGHT_DECAY, device=device, patience=LSTM_PATIENCE,
        model_name="LSTM-MLP",
    )

    val_metrics = evaluate_model(model, X_val, y_val, model_name="LSTM-MLP", device=device)
    elapsed = time.time() - start

    save_dir = output_dir / "lstm_mlp"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim, "num_chunks": LSTM_NUM_CHUNKS,
        "lstm_hidden": LSTM_HIDDEN, "lstm_layers": LSTM_LAYERS,
        "method": "lstm_mlp_classifier",
    }, save_dir / "lstm_mlp_classifier.pt")
    print(f"  Saved to: {save_dir}")

    return model, val_metrics, elapsed


def run_bilstm(input_dim, X_train, y_train, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 3: BiLSTM CLASSIFIER (Supervised, Optimized)")
    print("=" * 70)
    start = time.time()

    model = BiLSTMClassifier(
        input_dim=input_dim, projection_dim=BILSTM_HIDDEN,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=BILSTM_DROPOUT,
    )
    print(f"  Parameters: {count_params(model):,}")

    model, best_thresh = train_supervised_weighted(
        model, X_train, y_train, X_val, y_val,
        epochs=BILSTM_EPOCHS, lr=BILSTM_LR, batch_size=BILSTM_BATCH_SIZE,
        weight_decay=BILSTM_WEIGHT_DECAY, device=device, patience=BILSTM_PATIENCE,
        model_name="BiLSTM",
        pos_weight=2.5,
        label_smoothing=0.05,
        warmup_epochs=3,
    )

    val_metrics = evaluate_model(model, X_val, y_val, model_name="BiLSTM", device=device, threshold=best_thresh)
    elapsed = time.time() - start

    save_dir = output_dir / "bilstm"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim, "projection_dim": BILSTM_HIDDEN,
        "lstm_hidden": LSTM_HIDDEN, "method": "bilstm_classifier",
    }, save_dir / "bilstm_classifier.pt")
    print(f"  Saved to: {save_dir}")

    return model, val_metrics, elapsed


def run_autoencoder(input_dim, X_train_normal, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 4: AUTOENCODER (Unsupervised Anomaly Detection)")
    print("=" * 70)
    start = time.time()

    model = AutoEncoder(input_dim=input_dim, latent_dim=AE_LATENT_DIM, dropout_rate=AE_DROPOUT)
    print(f"  Parameters: {count_params(model):,}")

    model = train_autoencoder(
        model, X_train_normal,
        epochs=AE_EPOCHS, lr=AE_LR, batch_size=AE_BATCH_SIZE,
        weight_decay=AE_WEIGHT_DECAY, device=device,
    )

    val_metrics = evaluate_model(
        model, X_val, y_val, model_name="AutoEncoder", device=device, is_autoencoder=True,
    )
    elapsed = time.time() - start

    save_dir = output_dir / "autoencoder"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim, "latent_dim": AE_LATENT_DIM,
        "dropout_rate": AE_DROPOUT, "threshold": val_metrics["threshold"],
        "method": "autoencoder",
    }, save_dir / "autoencoder.pt")
    print(f"  Saved to: {save_dir}")

    return model, val_metrics, elapsed


def run_transformer_encoder(input_dim, X_train, y_train, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 5: TRANSFORMER ENCODER CLASSIFIER (Supervised)")
    print("=" * 70)
    start = time.time()

    model = TransformerEncoderClassifier(
        input_dim=input_dim,
        num_chunks=TRANSFORMER_NUM_CHUNKS,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_DIM_FF,
        dropout=TRANSFORMER_DROPOUT,
        mlp_hidden=TRANSFORMER_MLP_HIDDEN,
        mlp_dropout=TRANSFORMER_MLP_DROPOUT,
    )
    print(f"  Parameters: {count_params(model):,}")

    model = train_supervised(
        model, X_train, y_train, X_val, y_val,
        epochs=TRANSFORMER_EPOCHS, lr=TRANSFORMER_LR, batch_size=TRANSFORMER_BATCH_SIZE,
        weight_decay=TRANSFORMER_WEIGHT_DECAY, device=device, patience=TRANSFORMER_PATIENCE,
        model_name="TransformerEncoder",
    )

    val_metrics = evaluate_model(model, X_val, y_val, model_name="TransformerEncoder", device=device)
    elapsed = time.time() - start

    save_dir = output_dir / "transformer_encoder"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "num_chunks": TRANSFORMER_NUM_CHUNKS,
        "d_model": TRANSFORMER_D_MODEL,
        "nhead": TRANSFORMER_NHEAD,
        "num_layers": TRANSFORMER_NUM_LAYERS,
        "dim_feedforward": TRANSFORMER_DIM_FF,
        "dropout": TRANSFORMER_DROPOUT,
        "mlp_hidden": TRANSFORMER_MLP_HIDDEN,
        "mlp_dropout": TRANSFORMER_MLP_DROPOUT,
        "method": "transformer_encoder_classifier",
    }, save_dir / "transformer_encoder_classifier.pt")
    print(f"  Saved to: {save_dir}")

    return model, val_metrics, elapsed


def run_tabnet(input_dim, X_train, y_train, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 6: TABNET CLASSIFIER (Supervised)")
    print("=" * 70)
    start = time.time()

    model = TabNetClassifier(
        input_dim=input_dim,
        n_steps=TABNET_N_STEPS,
        dims=TABNET_DIMS,
        sparse_sparsity=TABNET_SPARSE_SPARSITY,
        dropout=TABNET_DROPOUT,
    )
    print(f"  Parameters: {count_params(model):,}")

    model = train_supervised(
        model, X_train, y_train, X_val, y_val,
        epochs=TABNET_EPOCHS, lr=TABNET_LR, batch_size=TABNET_BATCH_SIZE,
        weight_decay=TABNET_WEIGHT_DECAY, device=device, patience=TABNET_PATIENCE,
        model_name="TabNet",
    )

    val_metrics = evaluate_model(model, X_val, y_val, model_name="TabNet", device=device)
    elapsed = time.time() - start

    save_dir = output_dir / "tabnet"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "n_steps": TABNET_N_STEPS,
        "dims": TABNET_DIMS,
        "sparse_sparsity": TABNET_SPARSE_SPARSITY,
        "dropout": TABNET_DROPOUT,
        "method": "tabnet_classifier",
    }, save_dir / "tabnet_classifier.pt")
    print(f"  Saved to: {save_dir}")

    return model, val_metrics, elapsed


def run_attention_mlp(input_dim, X_train, y_train, X_val, y_val, device, output_dir):
    print("\n" + "=" * 70)
    print("MODEL 7: ATTENTION-MLP CLASSIFIER (Optimized, Supervised)")
    print("=" * 70)
    start = time.time()

    model = AttentionMLPClassifier(
        input_dim=input_dim,
        num_chunks=ATTN_MLP_NUM_CHUNKS,
        d_model=ATTN_MLP_D_MODEL,
        nhead=ATTN_MLP_NHEAD,
        num_attn_layers=ATTN_MLP_NUM_LAYERS,
        dim_feedforward=ATTN_MLP_DIM_FF,
        dropout=ATTN_MLP_DROPOUT,
        stochastic_depth_rate=ATTN_MLP_STOCHASTIC_DEPTH,
        mlp_hidden=ATTN_MLP_HIDDEN,
        mlp_dropout=ATTN_MLP_DROPOUT_HEAD,
    )
    print(f"  Parameters: {count_params(model):,}")

    model, best_thresh = train_supervised_weighted(
        model, X_train, y_train, X_val, y_val,
        epochs=ATTN_MLP_EPOCHS, lr=ATTN_MLP_LR, batch_size=ATTN_MLP_BATCH_SIZE,
        weight_decay=ATTN_MLP_WEIGHT_DECAY, device=device, patience=ATTN_MLP_PATIENCE,
        model_name="Attention-MLP",
        pos_weight=ATTN_MLP_POS_WEIGHT,
        label_smoothing=ATTN_MLP_LABEL_SMOOTHING,
        warmup_epochs=ATTN_MLP_WARMUP,
    )

    val_metrics = evaluate_model(
        model, X_val, y_val, model_name="Attention-MLP", device=device, threshold=best_thresh
    )
    elapsed = time.time() - start

    save_dir = output_dir / "attention_mlp"
    ensure_dir(save_dir)
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "input_dim": input_dim,
        "num_chunks": ATTN_MLP_NUM_CHUNKS,
        "d_model": ATTN_MLP_D_MODEL,
        "nhead": ATTN_MLP_NHEAD,
        "num_attn_layers": ATTN_MLP_NUM_LAYERS,
        "dim_feedforward": ATTN_MLP_DIM_FF,
        "dropout": ATTN_MLP_DROPOUT,
        "stochastic_depth_rate": ATTN_MLP_STOCHASTIC_DEPTH,
        "mlp_hidden": ATTN_MLP_HIDDEN,
        "mlp_dropout": ATTN_MLP_DROPOUT_HEAD,
        "threshold": best_thresh,
        "method": "attention_mlp_classifier",
    }, save_dir / "attention_mlp_classifier.pt")
    print(f"  Saved to: {save_dir}")
    return model, val_metrics, elapsed


# ============================================================
# Test Evaluation
# ============================================================
def evaluate_on_test(model, X_test, y_test, device, model_name, is_ae=False, threshold=None):
    """Evaluate a trained model on the test set. Returns metrics with latency."""
    print(f"\n--- {model_name} on TEST SET ---")
    print(f"  Samples: {len(y_test)} | Normal: {(y_test==0).sum()} | Attack: {(y_test==1).sum()}")

    # Warm-up CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.time()
    if is_ae:
        scores = predict_autoencoder(model, X_test, device=device)
        if threshold is None:
            threshold, _ = find_best_threshold(scores, y_test)
        preds = (scores > threshold).astype(int)
    else:
        probs = predict_supervised(model, X_test, device=device)
        scores = probs
        preds = (probs >= 0.5).astype(int)
    elapsed = time.time() - start

    inference_ms = (elapsed / len(y_test)) * 1000  # ms per sample

    print(f"  Inference latency (batch): {elapsed:.3f}s total ({inference_ms:.2f} ms/sample)")

    metrics = evaluate_predictions(y_test, preds, scores, method_name=f"{model_name} (TEST)")
    metrics["inference_seconds"] = round(elapsed, 4)
    metrics["inference_ms_per_sample"] = round(inference_ms, 4)
    return metrics


def evaluate_end_to_end_latency(
    model,
    questions: list,
    answers: list,
    y_test: np.ndarray,
    device: str,
    model_name: str,
    scaler,
    is_ae: bool = False,
    threshold: float = None,
    max_samples: int = None,
    embedder: object = None,
):
    """
    End-to-end latency measurement: encode (batch_size=1) + feature fusion + inference (batch_size=1).

    Simulates production — one request at a time, no batching.

    Args:
        embedder: optional pre-loaded TextPairEmbedder to reuse across models.
                  If None, a new one is created.

    Returns:
        dict with avg/min/max/p50/p95/p99 latency in ms, accuracy, total_time, throughput
    """
    from src.utils.features import build_pair_features
    from src.data.dataset import FeatureDataset
    import statistics

    if embedder is None:
        from src.data.encoder import TextPairEmbedder
        embedder = TextPairEmbedder(device=device)

    # Limit samples for faster measurement
    n = len(questions) if max_samples is None else min(max_samples, len(questions))
    questions_sub = questions[:n]
    answers_sub = answers[:n]
    y_sub = y_test[:n]

    # Warm-up: run a few samples through the full pipeline
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    warmup = min(10, n)
    for _ in range(warmup):
        hq_w, ha_w = embedder.encode_pairs([questions_sub[0]], [answers_sub[0]], batch_size=1)
        z_w = build_pair_features(hq_w, ha_w)
        z_scaled = scaler.transform(z_w).astype(np.float32)
        _ = model(torch.tensor(z_scaled, device=device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # Measure latency per sample
    per_sample_latencies = []  # ms
    all_preds = []
    all_scores = []

    model.eval()
    model.to(device)

    print(f"  Measuring end-to-end latency ({n} samples, batch_size=1)...")
    for i in range(n):
        start = time.time()

        # Step 1: Encode doctor question + patient answer (batch_size=1)
        hq, ha = embedder.encode_pairs([questions_sub[i]], [answers_sub[i]], batch_size=1)

        # Step 2: Feature fusion + scaling
        z = build_pair_features(hq, ha)
        z_scaled = scaler.transform(z).astype(np.float32)

        # Step 3: Model inference (batch_size=1)
        with torch.no_grad():
            input_tensor = torch.tensor(z_scaled, device=device)
            if is_ae:
                recon = model(input_tensor)
                error = torch.mean((input_tensor - recon) ** 2, dim=0).item()
                score = error
                pred = 1 if score > threshold else 0
            else:
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).item()
                score = prob
                pred = 1 if prob >= 0.5 else 0

        if device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed_ms = (time.time() - start) * 1000
        per_sample_latencies.append(elapsed_ms)
        all_preds.append(pred)
        all_scores.append(score)

    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    # Statistics
    avg_ms = statistics.mean(per_sample_latencies)
    median_ms = statistics.median(per_sample_latencies)
    p95_ms = np.percentile(per_sample_latencies, 95)
    p99_ms = np.percentile(per_sample_latencies, 99)
    min_ms = min(per_sample_latencies)
    max_ms = max(per_sample_latencies)
    total_time = sum(per_sample_latencies) / 1000
    throughput = n / total_time if total_time > 0 else 0

    # Accuracy
    acc = accuracy_score(y_sub, all_preds) if len(y_sub) > 0 else 0

    print(f"    Avg: {avg_ms:.1f}ms | Median: {median_ms:.1f}ms | P95: {p95_ms:.1f}ms | P99: {p99_ms:.1f}ms")
    print(f"    Min: {min_ms:.1f}ms | Max: {max_ms:.1f}ms | Throughput: {throughput:.1f} samples/s")
    print(f"    Accuracy on {n} samples: {acc:.4f}")

    return {
        "samples": int(n),
        "total_time_seconds": round(total_time, 4),
        "avg_latency_ms": round(avg_ms, 4),
        "median_latency_ms": round(median_ms, 4),
        "p95_latency_ms": round(p95_ms, 4),
        "p99_latency_ms": round(p99_ms, 4),
        "min_latency_ms": round(min_ms, 4),
        "max_latency_ms": round(max_ms, 4),
        "throughput_samples_per_sec": round(throughput, 4),
        "accuracy": round(acc, 4),
    }


# ============================================================
# Summary Table
# ============================================================
def print_summary(val_results: list, test_results: list):
    print("\n" + "=" * 130)
    print("SUMMARY — Validation Results")
    print("=" * 130)
    header = (
        f"{'Method':<15} {'Params':>8} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'AtkRecall':>10} "
        f"{'AtkF2':>10} {'Time(s)':>8}"
    )
    print(header)
    print("-" * 130)
    for r in val_results:
        auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
        print(
            f"{r['method']:<15} {r['num_params']:>8,} "
            f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f} "
            f"{auc_str:>10} {r['attack_recall']:>10.4f} "
            f"{r['attack_f2']:>10.4f} {r['elapsed_seconds']:>8.0f}"
        )

    print("\n" + "=" * 130)
    print("SUMMARY — Test Set Results (MedAgentSim)")
    print("=" * 130)
    print(header)
    print("-" * 130)
    for r in test_results:
        auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
        print(
            f"{r['method']:<15} {r['num_params']:>8,} "
            f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f} "
            f"{auc_str:>10} {r['attack_recall']:>10.4f} "
            f"{r['attack_f2']:>10.4f} {'N/A':>8}"
        )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MedPIDefense Deep Learning Baseline")
    parser.add_argument(
        "--model",
        choices=["mlp", "lstm_mlp", "bilstm", "autoencoder", "transformer_encoder", "tabnet", "attention_mlp", "all"],
        default="all",
        help="Model(s) to train. Default: all",
    )
    parser.add_argument("--train_csv", type=str, default=None, help="Training CSV path")
    parser.add_argument("--test_csv", type=str, default=None, help="Test CSV path")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument(
        "--test-only", action="store_true",
        help="Skip training, evaluate trained models on test set only",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    train_csv = args.train_csv or str(TRAIN_CSV)
    test_csv = args.test_csv or str(TEST_CSV)
    output_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR
    ensure_dir(output_dir)

    # ============================================================
    # Load Test Data (always needed for evaluation)
    # ============================================================
    print("\n=== Loading Test Data ===")
    df_test = load_csv(test_csv)
    print(f"  Total: {len(df_test)} | Normal: {(df_test['label']==0).sum()} | Attack: {(df_test['label']==1).sum()}")

    # ============================================================
    # Data Pipeline: Train mode vs Test-only mode
    # ============================================================
    emb_dir = output_dir / "embeddings"
    import pickle

    if not args.test_only:
        # --- TRAINING MODE: load train data, encode, prepare features ---
        print("\n=== Loading Training Data ===")
        df_train = load_csv(train_csv)
        print(f"  Total: {len(df_train)} | Normal: {(df_train['label']==0).sum()} | Attack: {(df_train['label']==1).sum()}")

        print("\n=== Encoding with BiomedBERT ===")
        embedder = TextPairEmbedder(device=device)

        print("  Encoding training set...")
        hq_train, ha_train = embedder.encode_pairs(
            df_train["doctor_question"].tolist(),
            df_train["patient_answer"].tolist(),
            batch_size=EMBED_BATCH_SIZE,
        )

        print("  Encoding test set...")
        hq_test, ha_test = embedder.encode_pairs(
            df_test["doctor_question"].tolist(),
            df_test["patient_answer"].tolist(),
            batch_size=EMBED_BATCH_SIZE,
        )

        # Prepare Features — Supervised Split
        sup_train_idx, sup_val_idx = split_supervised(df_train, train_ratio=SUP_TRAIN_RATIO, seed=args.seed)
        df_sup_train = df_train.iloc[sup_train_idx]
        df_sup_val = df_train.iloc[sup_val_idx]

        X_sup_train, scaler_sup = prepare_features(df_sup_train, hq_train, ha_train, fit=True)
        X_sup_val, _ = prepare_features(df_sup_val, hq_train, ha_train, scaler=scaler_sup, fit=False)

        # save the scaler for supervised features
        scaler_sup_path = output_dir / "scaler_sup.pkl"
        with open(scaler_sup_path, "wb") as f:
            pickle.dump(scaler_sup, f)

        # save embeddings for test-only mode (avoid re-encoding)
        ensure_dir(emb_dir)
        np.save(str(emb_dir / "hq_train.npy"), hq_train)
        np.save(str(emb_dir / "ha_train.npy"), ha_train)
        np.save(str(emb_dir / "hq_test.npy"), hq_test)
        np.save(str(emb_dir / "ha_test.npy"), ha_test)
        print(f"  Saved embeddings to {emb_dir}")

        y_sup_train = df_sup_train["label"].values.astype(int)
        y_sup_val = df_sup_val["label"].values.astype(int)
        input_dim = X_sup_train.shape[1]

        print(f"  Supervised Train: {X_sup_train.shape} | Val: {X_sup_val.shape}")
        print(f"  Input dim: {input_dim}")

        # Prepare Features — AutoEncoder Split
        ae_train_idx, ae_val_idx = split_autoencoder(
            df_train, train_normal_ratio=AE_TRAIN_NORMAL_RATIO, seed=args.seed,
        )
        df_ae_train = df_train.iloc[ae_train_idx]
        df_ae_val = df_train.iloc[ae_val_idx]

        X_ae_train, scaler_ae = prepare_features(df_ae_train, hq_train, ha_train, fit=True)
        X_ae_val, _ = prepare_features(df_ae_val, hq_train, ha_train, scaler=scaler_ae, fit=False)

        # save the scaler for AE features
        scaler_ae_path = output_dir / "scaler_ae.pkl"
        with open(scaler_ae_path, "wb") as f:
            pickle.dump(scaler_ae, f)

        y_ae_val = df_ae_val["label"].values.astype(int)
        print(f"  AE Train (normal): {X_ae_train.shape} | AE Val: {X_ae_val.shape}")

        # Prepare Test Features (use supervised scaler for consistency)
        X_test, _ = prepare_features(df_test, hq_test, ha_test, scaler=scaler_sup, fit=False)
        y_test = df_test["label"].values.astype(int)
        print(f"  Test features: {X_test.shape}")

        # save test features for test-only mode
        np.save(str(emb_dir / "X_test.npy"), X_test)
        np.save(str(emb_dir / "y_test.npy"), y_test)

        # ============================================================
        # Train Models
        # ============================================================
        models = {}
        val_results = []
        ae_threshold = None

        model_trainers = {
            "mlp": lambda: run_mlp(input_dim, X_sup_train, y_sup_train, X_sup_val, y_sup_val, device, output_dir),
            "lstm_mlp": lambda: run_lstm_mlp(input_dim, X_sup_train, y_sup_train, X_sup_val, y_sup_val, device, output_dir),
            "bilstm": lambda: run_bilstm(input_dim, X_sup_train, y_sup_train, X_sup_val, y_sup_val, device, output_dir),
            "autoencoder": lambda: run_autoencoder(input_dim, X_ae_train, X_ae_val, y_ae_val, device, output_dir),
            "transformer_encoder": lambda: run_transformer_encoder(input_dim, X_sup_train, y_sup_train, X_sup_val, y_sup_val, device, output_dir),
            "tabnet": lambda: run_tabnet(input_dim, X_sup_train, y_sup_train, X_sup_val, y_sup_val, device, output_dir),
            "attention_mlp": lambda: run_attention_mlp(input_dim, X_sup_train, y_sup_train, X_sup_val, y_sup_val, device, output_dir),
        }

        keys = list(model_trainers.keys()) if args.model == "all" else [args.model]

        for key in keys:
            model, val_metrics, elapsed = model_trainers[key]()
            val_metrics["elapsed_seconds"] = round(elapsed, 1)
            val_metrics["num_params"] = count_params(model)
            val_results.append(val_metrics)
            models[key] = model
            if key == "autoencoder":
                ae_threshold = val_metrics.get("threshold")
    else:
        data_load_start = time.time()
        # Load scalers from disk
        print("\n=== Loading Scalers and Embeddings (skipping BiomedBERT encoding) ===")
        scaler_sup_path = output_dir / "scaler_sup.pkl"

        if not scaler_sup_path.exists():
            raise FileNotFoundError(f"Saved scaler not found: {scaler_sup_path}. Run training first.")
        if not emb_dir.exists():
            raise FileNotFoundError(f"Saved embeddings not found: {emb_dir}. Run training first.")

        with open(scaler_sup_path, "rb") as f:
            scaler_sup = pickle.load(f)
        print(f"  Loaded scaler: {scaler_sup_path}")

        # Load embeddings
        hq_test = np.load(str(emb_dir / "hq_test.npy"))
        ha_test = np.load(str(emb_dir / "ha_test.npy"))
        print(f"  Loaded test embeddings: hq={hq_test.shape}, ha={ha_test.shape}")

        # Rebuild test features from embeddings + scaler
        X_test, _ = prepare_features(df_test, hq_test, ha_test, scaler=scaler_sup, fit=False)
        y_test = df_test["label"].values.astype(int)
        input_dim = X_test.shape[1]
        print(f"  Test features: {X_test.shape} | Input dim: {input_dim}")

        # Load trained models
        print("\n=== Loading Trained Models ===")
        ckpt_paths = load_trained_model(output_dir, device)
        models = {}
        val_results = []
        ae_threshold = None

        keys = list(ckpt_paths.keys()) if args.model == "all" else [args.model]

        for key in keys:
            ckpt = torch.load(ckpt_paths[key], map_location=device, weights_only=False)
            if key == "mlp":
                model = MLPClassifier(
                    input_dim=ckpt["input_dim"],
                    hidden_dims=ckpt["hidden_dims"], dropout=ckpt["dropout"],
                )
            elif key == "lstm_mlp":
                model = LSTMMLPClassifier(
                    input_dim=ckpt["input_dim"], num_chunks=ckpt["num_chunks"],
                    lstm_hidden=ckpt["lstm_hidden"], lstm_layers=ckpt["lstm_layers"],
                )
            elif key == "bilstm":
                model = BiLSTMClassifier(
                    input_dim=ckpt["input_dim"], projection_dim=ckpt["projection_dim"],
                    lstm_hidden=ckpt["lstm_hidden"],
                )
            elif key == "autoencoder":
                model = AutoEncoder(
                    input_dim=ckpt["input_dim"], latent_dim=ckpt["latent_dim"],
                    dropout_rate=ckpt["dropout_rate"],
                )
                ae_threshold = ckpt.get("threshold")
            elif key == "transformer_encoder":
                model = TransformerEncoderClassifier(
                    input_dim=ckpt["input_dim"],
                    num_chunks=ckpt["num_chunks"],
                    d_model=ckpt["d_model"],
                    nhead=ckpt["nhead"],
                    num_layers=ckpt["num_layers"],
                    dim_feedforward=ckpt["dim_feedforward"],
                    dropout=ckpt["dropout"],
                    mlp_hidden=ckpt["mlp_hidden"],
                    mlp_dropout=ckpt["mlp_dropout"],
                )
            elif key == "tabnet":
                model = TabNetClassifier(
                    input_dim=ckpt["input_dim"],
                    n_steps=ckpt["n_steps"],
                    dims=ckpt["dims"],
                    sparse_sparsity=ckpt["sparse_sparsity"],
                    dropout=ckpt["dropout"],
                )
            elif key == "attention_mlp":
                model = AttentionMLPClassifier(
                    input_dim=ckpt["input_dim"],
                    num_chunks=ckpt["num_chunks"],
                    d_model=ckpt["d_model"],
                    nhead=ckpt["nhead"],
                    num_attn_layers=ckpt["num_attn_layers"],
                    dim_feedforward=ckpt["dim_feedforward"],
                    dropout=ckpt["dropout"],
                    stochastic_depth_rate=ckpt["stochastic_depth_rate"],
                    mlp_hidden=ckpt["mlp_hidden"],
                    mlp_dropout=ckpt["mlp_dropout"],
                )
            else:
                continue
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            models[key] = model
            print(f"  Loaded: {key} ({count_params(model):,} params)")

        # placeholder variables for report (not needed in test-only)
        X_sup_train = X_sup_val = X_ae_train = X_ae_val = None

    # ============================================================
    # Evaluate on Test Set
    # ============================================================
    print("\n" + "#" * 70)
    print("# TEST SET EVALUATION")
    print("#" * 70)

    test_results = []
    for key, model in models.items():
        is_ae = key == "autoencoder"
        test_metrics = evaluate_on_test(
            model, X_test, y_test, device,
            model_name=key.upper(),
            is_ae=is_ae,
            threshold=ae_threshold if is_ae else None,
        )
        test_metrics["num_params"] = count_params(model)
        test_results.append(test_metrics)

    # ============================================================
    # End-to-End Latency Measurement (batch_size=1, production-like)
    # ============================================================
    e2e_results = {}
    if args.test_only:
        print("\n" + "#" * 70)
        print("# END-TO-END LATENCY (batch_size=1, encoding + inference)")
        print("#" * 70)

        # Load embedder once, reuse for all models
        embedder = TextPairEmbedder(device=device)

        # Only measure top models to save time
        top_models = ["attention_mlp", "mlp", "bilstm"]
        for key in top_models:
            if key not in models:
                continue
            model = models[key]
            is_ae = key == "autoencoder"
            e2e = evaluate_end_to_end_latency(
                model,
                df_test["doctor_question"].tolist(),
                df_test["patient_answer"].tolist(),
                y_test,
                device,
                model_name=key.upper(),
                scaler=scaler_sup,
                is_ae=is_ae,
                threshold=ae_threshold if is_ae else None,
                max_samples=100,  # 100 samples for latency estimation
                embedder=embedder,  # reuse the same embedder
            )
            e2e_results[key] = e2e

    # ============================================================
    # Print Summary
    # ============================================================
    if not args.test_only:
        print_summary(val_results, test_results)
    else:
        print("\n" + "=" * 185)
        print("TEST SET RESULTS (MedAgentSim)")
        print("=" * 185)
        header = (
            f"{'Method':<15} {'Params':>8} {'Accuracy':>10} {'Precision':>10} "
            f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'AtkRecall':>10} "
            f"{'AtkF2':>10} {'Batch(ms)':>10} {'E2E-Avg(ms)':>12} {'E2E-P95(ms)':>12}"
        )
        print(header)
        print("-" * 185)
        for r in test_results:
            auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
            ms_str = f"{r['inference_ms_per_sample']:>8.2f}" if r.get('inference_ms_per_sample') else "N/A"
            key_name = r['method'].replace(' (TEST)', '').lower()
            e2e = e2e_results.get(key_name)
            if e2e:
                e2e_avg = f"{e2e['avg_latency_ms']:>10.1f}"
                e2e_p95 = f"{e2e['p95_latency_ms']:>10.1f}"
            else:
                e2e_avg = "N/A"
                e2e_p95 = "N/A"
            print(
                f"{r['method']:<15} {r['num_params']:>8,} "
                f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} {r['f1']:>10.4f} "
                f"{auc_str:>10} {r['attack_recall']:>10.4f} "
                f"{r['attack_f2']:>10.4f} {ms_str:>10} {e2e_avg:>12} {e2e_p95:>12}"
            )

    # ============================================================
    # Save Full Results
    # ============================================================
    # Total data loading time for test-only mode
    data_load_time = None
    if args.test_only:
        data_load_time = round(time.time() - data_load_start, 4)
        print(f"\n  Data loading time (scalers + embeddings): {data_load_time}s")

    report = {
        "timestamp": datetime.now().isoformat(),
        "encoder_model": ENCODER_MODEL,
        "max_length": MAX_LENGTH,
        "pooling": POOLING,
        "seed": args.seed,
        "train_csv": train_csv,
        "test_csv": test_csv,
        "data_split": {
            "train_total": len(df_train) if not args.test_only else None,
            "train_normal": int((df_train["label"] == 0).sum()) if not args.test_only else None,
            "train_attack": int((df_train["label"] == 1).sum()) if not args.test_only else None,
            "test_total": len(df_test),
            "test_normal": int((df_test["label"] == 0).sum()),
            "test_attack": int((df_test["label"] == 1).sum()),
            "supervised_train": len(X_sup_train) if X_sup_train is not None else None,
            "supervised_val": len(X_sup_val) if X_sup_val is not None else None,
            "ae_train_normal": len(X_ae_train) if X_ae_train is not None else None,
            "ae_val": len(X_ae_val) if X_ae_val is not None else None,
        },
        "feature_dim": input_dim,
        "feature_formula": FEATURE_FORMULA,
        "data_loading_seconds": data_load_time,
        "end_to_end_latency": e2e_results if args.test_only else None,
        "validation_results": val_results if not args.test_only else None,
        "test_results": test_results,
    }

    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")
    return report


if __name__ == "__main__":
    main()
