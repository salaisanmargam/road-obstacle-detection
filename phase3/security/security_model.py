# phase3/security/security_model.py
"""
Network Anomaly Detector for Phase 3.

Primary model:   Random Forest (100 estimators, balanced class weights)
Baseline model:  Logistic Regression

Binary classification: BENIGN (0) vs ANOMALOUS (1)

Metrics reported:
  Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
  FPR, FNR, Confusion Matrix, Inference Latency
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

CKPT_DIR = Path("D:/road obstacle detection/road_damage_research/phase3/checkpoints")
RESULTS_DIR = Path("D:/road obstacle detection/road_damage_research/results/phase3")

SEED = 42


def train_random_forest(X_train, y_train):
    print("\n=== Training Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")
    return rf, train_time


def train_logistic_regression(X_train, y_train):
    print("\n=== Training Logistic Regression (baseline) ===")
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        solver="lbfgs",
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    lr.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")
    return lr, train_time


def evaluate_model(model, X, y, model_name: str, split_name: str = "test") -> dict:
    """Evaluate a trained model and return metrics dict."""
    # Predictions
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    pred_time = time.perf_counter() - t0
    latency_ms = (pred_time / len(X)) * 1000

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-9)

    # Core metrics
    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y, y_prob)
    except Exception:
        pr_auc = float("nan")

    # Confusion matrix → FPR, FNR
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
    else:
        tn = fp = fn = tp = 0
        fpr = fnr = float("nan")

    metrics = {
        "model":        model_name,
        "split":        split_name,
        "accuracy":     round(float(acc),  4),
        "precision":    round(float(prec), 4),
        "recall":       round(float(rec),  4),
        "f1":           round(float(f1),   4),
        "roc_auc":      round(float(roc_auc), 4),
        "pr_auc":       round(float(pr_auc),  4),
        "fpr":          round(float(fpr),  4),
        "fnr":          round(float(fnr),  4),
        "latency_ms_per_sample": round(float(latency_ms), 4),
        "confusion_matrix": cm.tolist(),
        "n_samples":    int(len(y)),
        "n_attack":     int(y.sum()),
        "n_benign":     int((y == 0).sum()),
    }

    print(f"\n  [{model_name}] {split_name} results:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    print(f"    PR-AUC:    {pr_auc:.4f}")
    print(f"    FPR:       {fpr:.4f}")
    print(f"    FNR:       {fnr:.4f}")
    print(f"    Latency:   {latency_ms:.4f} ms/sample")
    print(f"    CM:\n{cm}")

    return metrics


def save_model(model, model_name: str):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{model_name}.joblib"
    joblib.dump(model, path)
    print(f"  Saved model → {path}")
    return path


def run_security_training(X_train, y_train, X_val, y_val, X_test, y_test):
    """Full security model training + evaluation pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf, rf_train_time = train_random_forest(X_train, y_train)
    save_model(rf, "random_forest")

    rf_val_metrics  = evaluate_model(rf, X_val,  y_val,  "random_forest", "val")
    rf_test_metrics = evaluate_model(rf, X_test, y_test, "random_forest", "test")
    rf_test_metrics["train_time_s"] = round(rf_train_time, 2)
    all_metrics.extend([rf_val_metrics, rf_test_metrics])

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr, lr_train_time = train_logistic_regression(X_train, y_train)
    save_model(lr, "logistic_regression")

    lr_val_metrics  = evaluate_model(lr, X_val,  y_val,  "logistic_regression", "val")
    lr_test_metrics = evaluate_model(lr, X_test, y_test, "logistic_regression", "test")
    lr_test_metrics["train_time_s"] = round(lr_train_time, 2)
    all_metrics.extend([lr_val_metrics, lr_test_metrics])

    # ── Save all metrics ──────────────────────────────────────────────────────
    metrics_path = RESULTS_DIR / "security_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved security metrics → {metrics_path}")

    return rf, lr, rf_test_metrics, lr_test_metrics


if __name__ == "__main__":
    from phase3.security.ids_preprocessor import run_preprocessing
    X_tr, y_tr, X_val, y_val, X_te, y_te, feat_cols, scaler = run_preprocessing()
    run_security_training(X_tr, y_tr, X_val, y_val, X_te, y_te)
