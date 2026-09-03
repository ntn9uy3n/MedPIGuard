"""
Evaluation metrics and reporting utilities.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_confusion(y_true: np.ndarray, preds: np.ndarray) -> dict:
    """Compute TP, FP, TN, FN."""
    y_true = np.asarray(y_true).astype(int)
    preds = np.asarray(preds).astype(int)

    tp = int(np.sum((y_true == 1) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def compute_attack_metrics(confusion: dict) -> dict:
    """Compute per-class attack metrics from confusion matrix."""
    tp, fn, fp, tn = confusion["tp"], confusion["fn"], confusion["fp"], confusion["tn"]

    attack_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    attack_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    normal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    attack_f2 = (5 * tp) / (5 * tp + 4 * fn + fp) if (5 * tp + 4 * fn + fp) > 0 else 0.0

    return {
        "attack_recall": round(float(attack_recall), 4),
        "attack_precision": round(float(attack_precision), 4),
        "normal_recall": round(float(normal_recall), 4),
        "attack_f2": round(float(attack_f2), 4),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray = None,
    method_name: str = "",
) -> dict:
    """
    Full evaluation: accuracy, precision, recall, F1, ROC-AUC, per-class metrics.

    Returns a dict with all metrics (JSON-serializable).
    """
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )

    roc_auc = None
    if scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, scores)
        except ValueError:
            pass

    confusion = compute_confusion(y_true, preds)
    attack_metrics = compute_attack_metrics(confusion)

    metrics = {
        "method": method_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc else None,
        **confusion,
        **attack_metrics,
    }

    # Print report
    print(f"\n  === {method_name} Evaluation ===")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall (F1) : {recall:.4f}")
    print(f"  F1          : {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC     : {roc_auc:.4f}")
    print(f"  Attack Recall : {attack_metrics['attack_recall']:.4f}")
    print(f"  Attack Prec   : {attack_metrics['attack_precision']:.4f}")
    print(f"  Normal Recall : {attack_metrics['normal_recall']:.4f}")
    print(f"  Attack F2     : {attack_metrics['attack_f2']:.4f}")
    print()
    print(classification_report(y_true, preds, target_names=["normal", "attack"], zero_division=0))

    return metrics


def find_best_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple:
    """
    Find optimal threshold by maximizing F1 across percentiles.

    Returns:
        (best_threshold, best_f1)
    """
    thresholds = np.percentile(scores, np.arange(1, 99, 0.5))
    best_f1, best_thresh = -1.0, 0.5

    for t in thresholds:
        preds = (scores > t).astype(int)
        f1 = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )[2]
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1
