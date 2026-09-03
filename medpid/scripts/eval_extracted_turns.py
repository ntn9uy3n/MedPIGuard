#!/usr/bin/env python3
"""
Evaluate all 4 trained models on the extracted_dialogue_turns.csv test set.

This script:
1. Loads the new test CSV (columns: doctor, patient, label, turn_index)
2. Renames columns to match the expected schema (doctor_question, patient_answer)
3. Loads all 4 trained model checkpoints from models_output/
4. Encodes pairs with BiomedBERT
5. Evaluates each model and prints classification reports + summary table
6. Saves detailed results to models_output_extracted_turns/
"""

import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import *
from src.utils import set_seed, ensure_dir, get_device
from src.data.loader import load_csv, prepare_features, split_supervised, split_autoencoder
from src.data.encoder import TextPairEmbedder
from src.models.mlp import MLPClassifier
from src.models.lstm_mlp import LSTMMLPClassifier
from src.models.bilstm import BiLSTMClassifier
from src.models.autoencoder import AutoEncoder
from src.train import predict_supervised, predict_autoencoder
from src.utils.metrics import evaluate_predictions, find_best_threshold


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_model(key: str, model_dir: Path, device: str):
    """Load a trained model checkpoint."""
    ckpt_files = {
        "mlp": model_dir / "mlp" / "mlp_classifier.pt",
        "lstm_mlp": model_dir / "lstm_mlp" / "lstm_mlp_classifier.pt",
        "bilstm": model_dir / "bilstm" / "bilstm_classifier.pt",
        "autoencoder": model_dir / "autoencoder" / "autoencoder.pt",
    }
    ckpt = torch.load(ckpt_files[key], map_location=device)

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
    else:
        raise ValueError(f"Unknown model: {key}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    threshold = ckpt.get("threshold")
    print(f"  Loaded: {key} ({count_params(model):,} params)")
    return model, threshold


def evaluate_model_on_test(model, X_test, y_test, device, model_name,
                           is_ae=False, threshold=None):
    """Evaluate a model on the test set and return metrics + predictions."""
    print(f"\n--- {model_name} on TEST SET ---")
    print(f"  Samples: {len(y_test)} | Normal: {(y_test==0).sum()} | Attack: {(y_test==1).sum()}")

    if is_ae:
        scores = predict_autoencoder(model, X_test, device=device)
        if threshold is None:
            threshold, _ = find_best_threshold(scores, y_test)
        preds = (scores > threshold).astype(int)
    else:
        probs = predict_supervised(model, X_test, device=device)
        scores = probs
        preds = (probs >= 0.5).astype(int)

    metrics = evaluate_predictions(y_test, preds, scores, method_name=model_name)
    return metrics, preds, scores, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    project_root = Path(__file__).resolve().parents[1]  # MedPIDefense_Proposal/
    model_dir = RESULTS_DIR / "models_output"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "extracted_turns"
    ensure_dir(output_dir)

    # ---- Load Training CSV (for scaler fitting) ----
    train_csv = DATA_DIR / "trainset.csv"
    print("\n=== Loading Training Data (for scaler) ===")
    df_train = load_csv(str(train_csv))
    print(f"  Total: {len(df_train)} | Normal: {(df_train['label']==0).sum()} | Attack: {(df_train['label']==1).sum()}")

    # ---- Load Test CSV ----
    test_csv_path = args.test_csv or str(DATA_DIR / "extracted_turns" / "extracted_dialogue_turns.csv")
    print(f"\n=== Loading Test Data ===")
    print(f"  Path: {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)
    print(f"  Columns: {list(df_test.columns)}")

    # Rename columns to match expected schema
    col_mapping = {}
    if "doctor" in df_test.columns and "doctor_question" not in df_test.columns:
        col_mapping["doctor"] = "doctor_question"
    if "patient" in df_test.columns and "patient_answer" not in df_test.columns:
        col_mapping["patient"] = "patient_answer"
    if col_mapping:
        df_test = df_test.rename(columns=col_mapping)
        print(f"  Renamed columns: {col_mapping}")

    df_test["doctor_question"] = df_test["doctor_question"].fillna("").astype(str)
    df_test["patient_answer"] = df_test["patient_answer"].fillna("").astype(str)
    df_test["label"] = df_test["label"].astype(int)

    print(f"  Total: {len(df_test)} | Normal: {(df_test['label']==0).sum()} | Attack: {(df_test['label']==1).sum()}")

    # ---- Encode with BiomedBERT ----
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

    # ---- Prepare Features (use supervised scaler from training) ----
    # Fit scaler on training data first
    from src.data.loader import split_supervised
    sup_train_idx, sup_val_idx = split_supervised(df_train, train_ratio=SUP_TRAIN_RATIO, seed=args.seed)
    df_sup_train = df_train.iloc[sup_train_idx]
    X_sup_train, scaler_sup = prepare_features(df_sup_train, hq_train, ha_train, fit=True)

    # Transform test data
    X_test, _ = prepare_features(df_test, hq_test, ha_test, scaler=scaler_sup, fit=False)
    y_test = df_test["label"].values.astype(int)
    print(f"  Test features: {X_test.shape}")

    # ---- Load Models ----
    print("\n=== Loading Trained Models ===")
    models = {}
    ae_threshold = None

    for key in ["mlp", "lstm_mlp", "bilstm", "autoencoder"]:
        model, threshold = load_model(key, model_dir, device)
        models[key] = model
        if key == "autoencoder":
            ae_threshold = threshold

    # ---- Evaluate ----
    print("\n" + "#" * 70)
    print("# TEST SET EVALUATION (extracted_dialogue_turns.csv)")
    print("#" * 70)

    test_results = []
    all_preds = {}

    for key, model in models.items():
        is_ae = key == "autoencoder"
        t0 = time.time()
        metrics, preds, scores, thresh = evaluate_model_on_test(
            model, X_test, y_test, device,
            model_name=key.upper(),
            is_ae=is_ae,
            threshold=ae_threshold if is_ae else None,
        )
        elapsed = time.time() - t0
        metrics["num_params"] = count_params(model)
        metrics["elapsed_seconds"] = round(elapsed, 1)
        if is_ae:
            metrics["threshold"] = float(thresh)
        test_results.append(metrics)
        all_preds[key] = preds
        print(f"  => {key.upper()}: F1={metrics['f1']:.4f}, AtkRecall={metrics['attack_recall']:.4f}, AtkF2={metrics['attack_f2']:.4f}")

    # ---- Print Summary ----
    print("\n" + "=" * 130)
    print("SUMMARY — Test Set Results (extracted_dialogue_turns.csv)")
    print("=" * 130)
    header = (
        f"{'Method':<15} {'Params':>8} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'AtkRecall':>10} "
        f"{'AtkF2':>10} {'Time(s)':>8}"
    )
    print(header)
    print("-" * 130)
    for r in test_results:
        auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
        print(
            f"{r['method']:<15} {r['num_params']:>8,} "
            f"{r['accuracy']:>10.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f} "
            f"{auc_str:>10} {r['attack_recall']:>10.4f} "
            f"{r['attack_f2']:>10.4f} {r['elapsed_seconds']:>8.0f}"
        )

    # ---- Save Results ----
    report = {
        "timestamp": datetime.now().isoformat(),
        "encoder_model": ENCODER_MODEL,
        "max_length": MAX_LENGTH,
        "pooling": POOLING,
        "seed": args.seed,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv_path),
        "test_data_info": {
            "total": int(len(df_test)),
            "normal": int((df_test["label"] == 0).sum()),
            "attack": int((df_test["label"] == 1).sum()),
            "turn_indices": sorted(df_test["turn_index"].unique().tolist()) if "turn_index" in df_test.columns else [],
        },
        "feature_dim": int(X_sup_train.shape[1]),
        "feature_formula": FEATURE_FORMULA,
        "test_results": test_results,
    }

    results_path = output_dir / "baseline_results_extracted_turns.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save per-sample predictions
    pred_df = pd.DataFrame({
        "doctor_question": df_test["doctor_question"].values,
        "patient_answer": df_test["patient_answer"].values,
        "label": y_test,
    })
    for key in all_preds:
        pred_df[f"pred_{key}"] = all_preds[key]
    pred_df.to_csv(output_dir / "predictions_extracted_turns.csv", index=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Predictions saved to: {output_dir / 'predictions_extracted_turns.csv'}")
    return report


if __name__ == "__main__":
    main()
