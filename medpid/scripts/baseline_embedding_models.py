#!/usr/bin/env python3
"""
Baseline: Different Embeddings + Small LLMs as Encoders
========================================================

Experiments:
  A) MLP Classifier with embedding backbones:
     Medical BERTs:
       - BiomedBERT (original baseline, 768d)
       - Bio_ClinicalBERT (768d)
       - PubMedBERT-abstract (768d)
       - PubMedBERT-abstract-fulltext (768d)
       - BioBERT-v1.1 (768d)
     General-purpose embeddings:
       - MiniLM-L6-v2 (384d)
       - MPNet-base-v2 (768d)
       - BGE-base-en-v1.5 (768d)
       - BGE-small-en-v1.5 (384d)
       - MXBAI-embed-large-v1 (1024d)
       - DistilRoBERTa-v1 (768d)
       - Nomic-embed-text-v1 (768d)
       - GTE-base-en-v1.5 (768d)
       - GTE-large (1024d)

  B) Small LLMs as Embedding Encoders (CLS pooling -> pair features -> MLP):
     - SmolLM2-135M (576d)
     - SmolLM2-360M (960d)
     - Qwen2.5-0.5B (896d)
     - Qwen2.5-1.5B (1536d)
     - TinyLlama-1.1B (2048d)

Usage:
    # Run all (Section A + B)
    PYTHONPATH=. python scripts/baseline_embedding_models.py --section all

    # Run only medical BERT embeddings
    PYTHONPATH=. python scripts/baseline_embedding_models.py --section embeddings --embed biomedbert clinicalbert pubmedbert pubmedbert_af biobert

    # Run only small LLM embeddings
    PYTHONPATH=. python scripts/baseline_embedding_models.py --section llms

    # Run specific embeddings
    PYTHONPATH=. python scripts/baseline_embedding_models.py --section embeddings --embed biomedbert bge_base smollm2_135m

    # Custom test CSV
    PYTHONPATH=. python scripts/baseline_embedding_models.py --test_csv data/extracted_turns/extracted_dialogue_turns.csv
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_fscore_support, roc_auc_score,
)
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
SEED = 42

# --- Medical BERTs ---
MEDICAL_BERTS = {
    "biomedbert": {
        "name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "dim": 768,
        "desc": "BiomedBERT (original baseline)",
        "category": "medical",
    },
    "clinicalbert": {
        "name": "emilyalsentzer/Bio_ClinicalBERT",
        "dim": 768,
        "desc": "Bio_ClinicalBERT",
        "category": "medical",
    },
    "pubmedbert": {
        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "dim": 768,
        "desc": "PubMedBERT-abstract",
        "category": "medical",
    },
    "pubmedbert_af": {
        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "dim": 768,
        "desc": "PubMedBERT-abstract+fulltext",
        "category": "medical",
    },
    "biobert": {
        "name": "dmis-lab/biobert-v1.1",
        "dim": 768,
        "desc": "BioBERT-v1.1",
        "category": "medical",
    },
}

# --- General-purpose embeddings ---
GENERAL_EMBEDDINGS = {
    "minilm_l6": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "desc": "MiniLM-L6-v2",
        "category": "general",
    },
    "mpnet_base": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768,
        "desc": "MPNet-base-v2",
        "category": "general",
    },
    "bge_base": {
        "name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "desc": "BGE-base-en-v1.5",
        "category": "general",
    },
    "bge_small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "desc": "BGE-small-en-v1.5",
        "category": "general",
    },
    "mxbai_large": {
        "name": "mixedbread-ai/mxbai-embed-large-v1",
        "dim": 1024,
        "desc": "MXBAI-embed-large-v1",
        "category": "general",
    },
    "distilroberta": {
        "name": "sentence-transformers/all-distilroberta-v1",
        "dim": 768,
        "desc": "DistilRoBERTa-v1",
        "category": "general",
    },
    "nomic_embed": {
        "name": "nomic-ai/nomic-embed-text-v1",
        "dim": 768,
        "desc": "Nomic-embed-text-v1",
        "category": "general",
    },
    # "gte_base": {
    #     "name": "Alibaba-NLP/gte-base-en-v1.5",
    #     "dim": 768,
    #     "desc": "GTE-base-en-v1.5",
    #     "category": "general",
    # },
    # "gte_large": {
    #     "name": "thenlper/gte-large",
    #     "dim": 1024,
    #     "desc": "GTE-large",
    #     "category": "general",
    # },
}

# --- Small LLMs as embedding encoders ---
SMALL_LLM_EMBEDDERS = {
    "smollm2_135m": {
        "name": "HuggingFaceTB/SmolLM2-135M",
        "dim": 576,
        "desc": "SmolLM2-135M (embedding)",
        "category": "llm_embed",
    },
    "smollm2_360m": {
        "name": "HuggingFaceTB/SmolLM2-360M",
        "dim": 960,
        "desc": "SmolLM2-360M (embedding)",
        "category": "llm_embed",
    },
    "qwen25_05b": {
        "name": "Qwen/Qwen2.5-0.5B",
        "dim": 896,
        "desc": "Qwen2.5-0.5B (embedding)",
        "category": "llm_embed",
    },
    "qwen25_15b": {
        "name": "Qwen/Qwen2.5-1.5B",
        "dim": 1536,
        "desc": "Qwen2.5-1.5B (embedding)",
        "category": "llm_embed",
    },
    "tinyllama_11b": {
        "name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "dim": 2048,
        "desc": "TinyLlama-1.1B (embedding)",
        "category": "llm_embed",
    },
}

# All embeddings combined
ALL_EMBEDDINGS = {**MEDICAL_BERTS, **GENERAL_EMBEDDINGS, **SMALL_LLM_EMBEDDERS}

MAX_LENGTH = 256
EMBED_BATCH_SIZE = 32
MLP_HIDDEN_DIMS = [1024, 512, 256]
MLP_DROPOUT = 0.3
MLP_EPOCHS = 50
MLP_LR = 1e-3
MLP_BATCH_SIZE = 32
MLP_PATIENCE = 10


# ============================================================
# Helpers
# ============================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Data Loading
# ============================================================
def load_and_prepare_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV, auto-map column names."""
    df = pd.read_csv(csv_path)
    col_map = {}
    if "doctor" in df.columns and "doctor_question" not in df.columns:
        col_map["doctor"] = "doctor_question"
    if "patient" in df.columns and "patient_answer" not in df.columns:
        col_map["patient"] = "patient_answer"
    if col_map:
        df = df.rename(columns=col_map)
    for col in ["doctor_question", "patient_answer", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in {csv_path}")
    df["doctor_question"] = df["doctor_question"].fillna("").astype(str)
    df["patient_answer"] = df["patient_answer"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


# ============================================================
# Embedding Encoder (supports BERT and small LLM architectures)
# ============================================================
class Embedder:
    """Encode texts using CLS pooling from any transformer model."""

    def __init__(self, model_name: str, device: str, max_length: int = 256):
        print(f"  Loading encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

        # Handle tokenizer edge cases for small LLMs
        if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = '[PAD]'

    @torch.no_grad()
    def encode(self, texts: list, batch_size: int = 32) -> np.ndarray:
        all_emb = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[start:start + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS / first token
            all_emb.append(emb.detach().cpu().numpy())
        return np.concatenate(all_emb, axis=0)


# ============================================================
# Feature Fusion
# ============================================================
def build_pair_features(hq: np.ndarray, ha: np.ndarray) -> np.ndarray:
    """z = [hq; ha; |hq-ha|; hq*ha; cos(hq,ha)] -> [N, 4*D + 1]"""
    diff = np.abs(hq - ha)
    prod = hq * ha
    cos_num = np.sum(hq * ha, axis=1, keepdims=True)
    cos_den = np.linalg.norm(hq, axis=1, keepdims=True) * np.linalg.norm(ha, axis=1, keepdims=True) + 1e-8
    cos = cos_num / cos_den
    return np.concatenate([hq, ha, diff, prod, cos], axis=1).astype(np.float32)


# ============================================================
# MLP Classifier
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h), nn.LayerNorm(h),
                nn.GELU(), nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ============================================================
# Training
# ============================================================
class FeatureDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def train_mlp(X_train, y_train, X_val, y_val, input_dim, device, label="MLP"):
    """Train MLP with early stopping on val F1."""
    model = SimpleMLP(input_dim, MLP_HIDDEN_DIMS, MLP_DROPOUT).to(device)
    print(f"  {label}: params={count_params(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MLP_EPOCHS, eta_min=1e-6)
    train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=MLP_BATCH_SIZE, shuffle=True)

    best_f1, best_state, wait = -1.0, None, 0
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            loss = criterion(model(features), labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32).to(device))).cpu().numpy()
        val_preds = (val_probs >= 0.5).astype(int)
        val_f1 = precision_recall_fscore_support(y_val, val_preds, average="binary", zero_division=0)[2]
        val_acc = accuracy_score(y_val, val_preds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{label}] Ep {epoch}/{MLP_EPOCHS} | Loss: {np.mean(losses):.6f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= MLP_PATIENCE:
                print(f"    [{label}] Early stop ep {epoch} (best F1={best_f1:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return model, best_f1


@torch.no_grad()
def predict_mlp(model, X, device, batch_size=64):
    model.eval()
    loader = DataLoader(FeatureDataset(X), batch_size=batch_size, shuffle=False)
    all_probs = []
    for features in loader:
        probs = torch.sigmoid(model(features.to(device))).cpu().numpy()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)


def measure_inference_latency(embedder, model, X_test, device, num_warmup=3, num_runs=10):
    """
    Measure end-to-end inference latency per sample (ms).

    Stages measured:
      - encode_latency: time to embed a single pair (doctor + patient)
      - mlp_latency: time for MLP forward pass on one feature vector
      - total_latency: encode + build_features + mlp (end-to-end per sample)

    Returns dict with mean / median / p95 in milliseconds.
    """
    def percentile(data, p):
        return np.percentile(data, p)

    # --- Warmup (pin GPU, compile kernels) ---
    for _ in range(num_warmup):
        _ = embedder.encode(["warmup text"] * 2, batch_size=1)
        _ = predict_mlp(model, X_test[:1], device, batch_size=1)
        torch.cuda.synchronize()

    # --- Measure encoding latency (single pair, repeated) ---
    encode_times = []
    sample_texts = ["sample text for latency measurement"] * 2
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = embedder.encode(sample_texts, batch_size=1)
        torch.cuda.synchronize()
        encode_times.append((time.perf_counter() - t0) * 1000)  # ms

    # --- Measure MLP latency (single feature vector, repeated) ---
    mlp_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predict_mlp(model, X_test[:1], device, batch_size=1)
        torch.cuda.synchronize()
        mlp_times.append((time.perf_counter() - t0) * 1000)  # ms

    # --- End-to-end latency (encode pair + build features + mlp) ---
    e2e_times = []
    # pick a few real samples to measure full pipeline
    real_indices = list(range(min(20, len(X_test))))
    for idx in real_indices:
        # encode two texts
        t0 = time.perf_counter()
        _ = embedder.encode(["latency test text A", "latency test text B"], batch_size=1)
        torch.cuda.synchronize()
        encode_t = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        _ = predict_mlp(model, X_test[idx:idx+1], device, batch_size=1)
        torch.cuda.synchronize()
        mlp_t = (time.perf_counter() - t0) * 1000

        e2e_times.append(encode_t + mlp_t)

    latency = {
        "encode_ms_mean": round(np.mean(encode_times), 2),
        "encode_ms_median": round(np.median(encode_times), 2),
        "encode_ms_p95": round(percentile(encode_times, 95), 2),
        "mlp_ms_mean": round(np.mean(mlp_times), 2),
        "mlp_ms_median": round(np.median(mlp_times), 2),
        "mlp_ms_p95": round(percentile(mlp_times, 95), 2),
        "e2e_ms_mean": round(np.mean(e2e_times), 2),
        "e2e_ms_median": round(np.median(e2e_times), 2),
        "e2e_ms_p95": round(percentile(e2e_times, 95), 2),
    }
    return latency


# ============================================================
# Evaluation
# ============================================================
def compute_attack_metrics(tp, fn, fp, tn):
    ar = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ap = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    nr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    af2 = (5 * tp) / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) > 0 else 0.0
    return {"attack_recall": round(ar, 4), "attack_precision": round(ap, 4),
            "normal_recall": round(nr, 4), "attack_f2": round(af2, 4)}


def evaluate(y_true, preds, scores=None, method_name=""):
    y_true, preds = np.asarray(y_true).astype(int), np.asarray(preds).astype(int)
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    roc_auc = None
    if scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, scores)
        except ValueError:
            pass
    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    atk = compute_attack_metrics(tp, fn, fp, tn)

    print(f"\n  === {method_name} ===")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}" if roc_auc else "  ROC-AUC   : N/A")
    print(f"  AtkRecall : {atk['attack_recall']:.4f}")
    print(f"  AtkF2     : {atk['attack_f2']:.4f}")
    print()
    print(classification_report(y_true, preds, target_names=["normal", "attack"], zero_division=0))

    return {
        "method": method_name, "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4), "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc else None, **atk,
    }


# ============================================================
# Run Embedding Baselines
# ============================================================
def model_checkpoint_dir(output_dir: Path, key: str) -> Path:
    """Per-model directory: <output_dir>/models/<key>/"""
    d = output_dir / "models" / key
    ensure_dir(d)
    return d


def is_model_complete(checkpoint_dir: Path) -> bool:
    """Check whether a model was already trained to completion."""
    return (checkpoint_dir / "model.pt").exists() and (checkpoint_dir / "metrics.json").exists()


def save_checkpoint(checkpoint_dir: Path, key: str, model, metrics, scaler, cfg):
    """Save model weights, scaler, metrics, and config to disk."""
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    with open(checkpoint_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(checkpoint_dir / "scaler.pkl", "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({"key": key, "cfg": cfg, "saved_at": datetime.now().isoformat()}, f, indent=2)


def load_results_from_checkpoints(embed_keys: list, output_dir: Path) -> list:
    """Load metrics for models that were already saved."""
    results = []
    for key in embed_keys:
        checkpoint_dir = model_checkpoint_dir(output_dir, key)
        metrics_path = checkpoint_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            results.append(metrics)
            print(f"  ✓ Loaded cached results for {key} (F1={metrics.get('f1', '?')})")
    return results


def run_embedding_baselines(df_train, df_test, device, output_dir, selected_embeds=None):
    print("\n" + "=" * 80)
    print("SECTION A: EMBEDDING BASELINES (CLS pooling -> pair features -> MLP)")
    print("=" * 80)

    train_idx, val_idx = train_test_split(
        np.arange(len(df_train)), test_size=0.2, random_state=SEED,
        stratify=df_train["label"].values,
    )
    df_tr = df_train.iloc[train_idx].reset_index(drop=True)
    df_val = df_train.iloc[val_idx].reset_index(drop=True)

    embed_keys = selected_embeds if selected_embeds else list(ALL_EMBEDDINGS.keys())
    results = []

    # --- Check which models are already complete (skip them) ---
    skipped = []
    for key in embed_keys:
        checkpoint_dir = model_checkpoint_dir(output_dir, key)
        if is_model_complete(checkpoint_dir):
            skipped.append(key)

    if skipped:
        print(f"\n  ⏩ Already trained ({len(skipped)}): {', '.join(skipped)} — loading cached results…\n")
        results = load_results_from_checkpoints(skipped, output_dir)

    # --- Models that still need to run ---
    remaining = [k for k in embed_keys if k not in skipped]

    for key in remaining:
        if key not in ALL_EMBEDDINGS:
            print(f"  Unknown: {key}, skipping")
            continue

        cfg = ALL_EMBEDDINGS[key]
        cat_label = {"medical": "MEDICAL", "general": "GENERAL", "llm_embed": "LLM-EMBED"}.get(cfg.get("category", ""), "UNK")
        checkpoint_dir = model_checkpoint_dir(output_dir, key)

        print(f"\n{'─' * 60}")
        print(f"  [{cat_label}] {key} — {cfg['desc']} (dim={cfg['dim']})")
        print(f"{'─' * 60}")

        try:
            embedder = Embedder(cfg["name"], device, max_length=MAX_LENGTH)

            hq_tr = embedder.encode(df_tr["doctor_question"].tolist(), EMBED_BATCH_SIZE)
            ha_tr = embedder.encode(df_tr["patient_answer"].tolist(), EMBED_BATCH_SIZE)
            hq_val = embedder.encode(df_val["doctor_question"].tolist(), EMBED_BATCH_SIZE)
            ha_val = embedder.encode(df_val["patient_answer"].tolist(), EMBED_BATCH_SIZE)
            hq_test = embedder.encode(df_test["doctor_question"].tolist(), EMBED_BATCH_SIZE)
            ha_test = embedder.encode(df_test["patient_answer"].tolist(), EMBED_BATCH_SIZE)

            X_tr = build_pair_features(hq_tr, ha_tr)
            X_val = build_pair_features(hq_val, ha_val)
            X_test = build_pair_features(hq_test, ha_test)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
            X_val_s = scaler.transform(X_val).astype(np.float32)
            X_test_s = scaler.transform(X_test).astype(np.float32)

            print(f"  Features: {X_tr_s.shape[1]}d (from {cfg['dim']}d embed)")

            t0 = time.time()
            model, val_f1 = train_mlp(X_tr_s, df_tr["label"].values, X_val_s, df_val["label"].values,
                                      X_tr_s.shape[1], device, label=key)
            train_time = time.time() - t0

            probs = predict_mlp(model, X_test_s, device)
            preds = (probs >= 0.5).astype(int)
            metrics = evaluate(df_test["label"].values, preds, scores=probs, method_name=f"MLP+{key}")
            metrics["embedding"] = key
            metrics["category"] = cfg.get("category", "unknown")
            metrics["embedding_dim"] = cfg["dim"]
            metrics["feature_dim"] = int(X_tr_s.shape[1])
            metrics["params"] = count_params(model)
            metrics["val_f1"] = round(val_f1, 4)
            metrics["train_time"] = round(train_time, 1)

            # --- Measure inference latency ---
            print(f"  ⏱ Measuring latency…")
            latency = measure_inference_latency(embedder, model, X_test_s, device)
            metrics.update(latency)
            print(f"    Encode: {latency['encode_ms_mean']:.1f}ms (mean) / {latency['encode_ms_p95']:.1f}ms (p95)")
            print(f"    MLP:    {latency['mlp_ms_mean']:.3f}ms (mean) / {latency['mlp_ms_p95']:.3f}ms (p95)")
            print(f"    E2E:    {latency['e2e_ms_mean']:.1f}ms (mean) / {latency['e2e_ms_p95']:.1f}ms (p95)")

            results.append(metrics)

            # --- Save checkpoint immediately after successful run ---
            save_checkpoint(checkpoint_dir, key, model, metrics, scaler, cfg)
            print(f"  💾 Checkpoint saved: {checkpoint_dir}")

            print(f"  => {key}: Val F1={val_f1:.4f}, Test F1={metrics['f1']:.4f}, AtkF2={metrics['attack_f2']:.4f}")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            error_entry = {"method": f"MLP+{key}", "embedding": key, "category": cfg.get("category", ""), "error": str(e)}
            results.append(error_entry)

        # Free memory
        try:
            del embedder, model, hq_tr, ha_tr, hq_val, ha_val, hq_test, ha_test
        except NameError:
            pass
        torch.cuda.empty_cache()

    return results


# ============================================================
# Summary Table
# ============================================================
def print_summary(results, title="Results"):
    print(f"\n{'=' * 130}")
    print(f"{title}")
    print(f"{'=' * 130}")
    header = (
        f"{'Method':<28} {'Category':<10} {'EmDim':>6} {'FeatDim':>8} "
        f"{'ValF1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} "
        f"{'AUC':>8} {'AtkR':>8} {'AtkF2':>8} {'Enc(ms)':>10} "
        f"{'E2E(ms)':>10} {'Params':>12} {'Time':>6}"
    )
    print(header)
    print("-" * 150)
    for r in results:
        if "error" in r:
            print(f"{r.get('method', 'ERR'):<28} ERROR: {r['error'][:60]}")
            continue
        cat = r.get("category", "")[:10]
        emd = r.get("embedding_dim", "?")
        fd = r.get("feature_dim", "?")
        vf1 = f"{r.get('val_f1', 0):.4f}"
        auc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
        params = f"{r.get('params', 0):,}"
        t = f"{r.get('train_time', '?')}"
        enc_ms = f"{r.get('encode_ms_mean', '?'):.1f}" if isinstance(r.get('encode_ms_mean'), (int, float)) else "?"
        e2e_ms = f"{r.get('e2e_ms_mean', '?'):.1f}" if isinstance(r.get('e2e_ms_mean'), (int, float)) else "?"
        print(
            f"{r['method']:<28} {cat:<10} {emd:>6} {fd:>8} "
            f"{vf1:>8} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1']:>8.4f} {auc:>8} "
            f"{r['attack_recall']:>8.4f} {r['attack_f2']:>8.4f} "
            f"{enc_ms:>10} {e2e_ms:>10} {params:>12} {t:>6}"
        )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Baseline: Embeddings + Small LLM Encoders")
    parser.add_argument("--section", choices=["embeddings", "all"], default="all")
    parser.add_argument("--embed", nargs="+", default=None, help="Specific embeddings to run (key names)")
    parser.add_argument("--category", choices=["medical", "general", "llm_embed", "all"], default="all",
                        help="Run only a category of embeddings")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "embedding_baselines"
    ensure_dir(output_dir)

    train_csv = Path(args.train_csv) if args.train_csv else data_dir / "trainset.csv"
    test_csv = Path(args.test_csv) if args.test_csv else data_dir / "extracted_turns" / "extracted_dialogue_turns.csv"

    print("\n=== Loading Data ===")
    df_train = load_and_prepare_csv(str(train_csv))
    df_test = load_and_prepare_csv(str(test_csv))
    print(f"  Train: {len(df_train)} (N={(df_train['label']==0).sum()}, A={(df_train['label']==1).sum()})")
    print(f"  Test:  {len(df_test)} (N={(df_test['label']==0).sum()}, A={(df_test['label']==1).sum()})")

    # Determine which embeddings to run
    if args.embed:
        selected = args.embed
    elif args.category != "all":
        selected = [k for k, v in ALL_EMBEDDINGS.items() if v.get("category") == args.category]
    else:
        selected = list(ALL_EMBEDDINGS.keys())

    print(f"\n  Selected embeddings ({len(selected)}): {', '.join(selected)}")

    # Run
    results = run_embedding_baselines(df_train, df_test, device, output_dir, selected_embeds=selected)
    print_summary(results, "RESULTS — Embedding Baselines (MLP Classifier)")

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_size": len(df_train),
        "test_size": len(df_test),
        "test_normal": int((df_test["label"] == 0).sum()),
        "test_attack": int((df_test["label"] == 1).sum()),
        "feature_formula": "[hq; ha; |hq-ha|; hq*ha; cos(hq,ha)]",
        "results": results,
    }
    results_path = output_dir / "baseline_embedding_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")
    return report


if __name__ == "__main__":
    main()
