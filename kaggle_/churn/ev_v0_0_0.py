import os
import json
import joblib
from datetime import datetime

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# =============================================================================
# ev_v0_0_0
# description: Initial utilities for evaluating churn prediction models.
# -----------------------------------------------------------------------------
# This module contains evaluation utilities for the churn prediction project.
# It centralizes metrics calculation, model saving, and registry logging.
# Functions herein are designed to ensure consistent and reproducible
# evaluation workflows for machine learning models.
# =============================================================================


def evaluate_model(model, X_train, y_train, X_test, y_test, mdl_name, fe_version, mdl_version, ev_version):
    # Predict (train)
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]

    # Predict (test)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "train": {
            "roc_auc": float(roc_auc_score(y_train, y_train_proba)),
            "precision": float(precision_score(y_train, y_train_pred, zero_division=0)),
            "recall": float(recall_score(y_train, y_train_pred, zero_division=0)),
            "f1": float(f1_score(y_train, y_train_pred, zero_division=0)),
        },
        "test": {
            "roc_auc": float(roc_auc_score(y_test, y_test_proba)),
            "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        },
    }

    print("Train Metrics")
    print(f"ROC AUC: {metrics['train']['roc_auc']:.4f}")

    print("\nTest Metrics")
    print(f"ROC AUC: {metrics['test']['roc_auc']:.4f}")

    # User inputs
    model_name = mdl_name

    feature_engineering_version = fe_version
    model_pipeline_version = mdl_version
    model_evaluation_version = ev_version

    # Version + paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"{model_name}_{timestamp}"
    package_dir = os.path.join("model_packages", model_version)
    os.makedirs(package_dir, exist_ok=True)

    # Save model package
    model_file = os.path.join(package_dir, f"{model_name}.joblib")
    joblib.dump(model, model_file)

    # Build log entry
    log_entry = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_version": model_version,
        "model_pipeline_version": model_pipeline_version,
        "feature_engineering_version": feature_engineering_version,
        "model_evaluation_version": model_evaluation_version,
        "model_file": model_file,
        "metrics": metrics,
        "features": list(X_train.columns),
    }

    # Append to JSON registry
    os.makedirs("logs", exist_ok=True)
    registry_file = os.path.join("logs", "model_registry.json")

    if os.path.exists(registry_file):
        with open(registry_file, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = []

    registry.append(log_entry)

    with open(registry_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"\nSaved model package: {model_file}")
    print(f"Updated registry: {registry_file}")