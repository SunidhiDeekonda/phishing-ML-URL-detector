from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float | int]:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    dataset_path = repo_root / "data" / "processed" / "dataset.csv"
    features_path = repo_root / "data" / "processed" / "features.csv"
    train_idx_path = repo_root / "data" / "processed" / "train_idx.npy"
    val_idx_path = repo_root / "data" / "processed" / "val_idx.npy"
    test_idx_path = repo_root / "data/processed/test_idx.npy"

    models_dir = repo_root / "models"
    results_dir = repo_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_model_path = models_dir / "lightgbm_raw.pkl"
    calibrated_model_path = models_dir / "lightgbm_calibrated.pkl"
    metrics_path = results_dir / "lightgbm_validation_metrics.json"
    predictions_path = results_dir / "lightgbm_validation_predictions.csv"
    importance_path = results_dir / "lightgbm_feature_importance.csv"

    dataset_df = pd.read_csv(dataset_path)
    features_df = pd.read_csv(features_path)

    if len(dataset_df) != 20_000:
        raise ValueError(f"Expected 20,000 rows in dataset.csv, found {len(dataset_df)}")

    if len(features_df) != 20_000:
        raise ValueError(f"Expected 20,000 feature rows, found {len(features_df)}")

    if features_df.shape[1] != 36:
        raise ValueError(f"Expected 36 engineered features, found {features_df.shape[1]}")

    if "label" not in dataset_df.columns:
        raise ValueError("dataset.csv must include a 'label' column")

    if dataset_df["label"].dtype == object:
        y = dataset_df["label"].astype(int).to_numpy()
    else:
        y = dataset_df["label"].to_numpy()

    # verify numeric-only feature set for LightGBM
    non_numeric_columns = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric_columns}")

    if any(col.lower() in {"url", "raw_url", "link"} for col in features_df.columns):
        raise ValueError("Feature set appears to include a URL/string column")

    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    if not (train_set.isdisjoint(val_set) and train_set.isdisjoint(test_set) and val_set.isdisjoint(test_set)):
        raise ValueError("Train/val/test index sets overlap")

    all_indices = set(range(len(dataset_df)))
    split_union = train_set | val_set | test_set
    if split_union != all_indices:
        raise ValueError("Train/val/test indices do not cover all dataset rows")

    if not np.issubdtype(train_idx.dtype, np.integer):
        raise TypeError("train_idx.npy must contain integer indices")

    if np.any((train_idx < 0) | (train_idx >= len(dataset_df))):
        raise ValueError("train_idx.npy contains out-of-range indices")
    if np.any((val_idx < 0) | (val_idx >= len(dataset_df))):
        raise ValueError("val_idx.npy contains out-of-range indices")
    if np.any((test_idx < 0) | (test_idx >= len(dataset_df))):
        raise ValueError("test_idx.npy contains out-of-range indices")

    X_train = features_df.iloc[train_idx]
    X_val = features_df.iloc[val_idx]
    y_train = y[train_idx].astype(int)
    y_val = y[val_idx].astype(int)

    if len(X_train) != 14_002:
        raise ValueError(f"Expected 14,002 training rows, found {len(X_train)}")

    if len(X_val) != 2_000:
        raise ValueError(f"Expected 2,000 validation rows, found {len(X_val)}")

    # label alignment sanity check
    if (y_train < 0).any() or (y_train > 1).any() or (y_val < 0).any() or (y_val > 1).any():
        raise ValueError("Labels must be binary (0/1)")

    model_config = {
        "objective": "binary",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = lgb.LGBMClassifier(**model_config)

    start_time = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )
    training_duration = time.perf_counter() - start_time

    raw_val_probability = model.predict_proba(X_val)[:, 1]
    raw_val_predictions = (raw_val_probability >= 0.5).astype(int)
    raw_val_metrics = evaluate_binary_metrics(y_val, raw_val_probability)

    frozen_model = FrozenEstimator(model)
    calibrator = CalibratedClassifierCV(
        estimator=frozen_model,
        method="sigmoid",
        cv=3,
    )
    calibrator.fit(X_val, y_val)

    calibrated_val_probability = calibrator.predict_proba(X_val)[:, 1]
    calibrated_val_predictions = (calibrated_val_probability >= 0.5).astype(int)
    calibrated_val_metrics = evaluate_binary_metrics(y_val, calibrated_val_probability)

    # sanity checks
    if len(raw_val_probability) != 2_000 or len(calibrated_val_probability) != 2_000:
        raise AssertionError("Validation prediction count must be 2,000")

    if not (np.isfinite(raw_val_probability).all() and np.isfinite(calibrated_val_probability).all()):
        raise AssertionError("Found non-finite probabilities")

    if np.any((raw_val_probability < 0) | (raw_val_probability > 1) | (calibrated_val_probability < 0) | (calibrated_val_probability > 1)):
        raise AssertionError("Validation probabilities must be in [0, 1]")

    if raw_val_metrics["tn"] + raw_val_metrics["fp"] + raw_val_metrics["fn"] + raw_val_metrics["tp"] != 2_000:
        raise AssertionError("Raw confusion matrix does not sum to 2,000")

    if calibrated_val_metrics["tn"] + calibrated_val_metrics["fp"] + calibrated_val_metrics["fn"] + calibrated_val_metrics["tp"] != 2_000:
        raise AssertionError("Calibrated confusion matrix does not sum to 2,000")

    # persist predictions
    predictions_df = pd.DataFrame(
        {
            "row_index": val_idx,
            "true_label": y_val,
            "raw_probability": raw_val_probability,
            "calibrated_probability": calibrated_val_probability,
            "raw_prediction": raw_val_predictions,
            "calibrated_prediction": calibrated_val_predictions,
        }
    )
    predictions_df.to_csv(predictions_path, index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": features_df.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    if len(feature_importance) != 36:
        raise AssertionError(f"Expected 36 feature importances, found {len(feature_importance)}")

    feature_importance.to_csv(importance_path, index=False)

    metrics_payload = {
        "lightgbm_version": lgb.__version__,
        "training_duration_seconds": training_duration,
        "num_training_rows": int(len(X_train)),
        "num_validation_rows": int(len(X_val)),
        "best_iteration": int(model.best_iteration_),
        "requested_n_estimators": model_config["n_estimators"],
        "early_stopping_triggered": bool(model.best_iteration_ < model_config["n_estimators"]),
        "raw_validation_metrics": raw_val_metrics,
        "calibrated_validation_metrics": calibrated_val_metrics,
        "top_10_features": feature_importance.head(10).to_dict(orient="records"),
    }

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)

    joblib.dump(model, raw_model_path)
    joblib.dump(calibrator, calibrated_model_path)

    # basic artifact presence checks
    missing = [
        p
        for p in [raw_model_path, calibrated_model_path, metrics_path, predictions_path, importance_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing expected artifact(s): {[str(p) for p in missing]}")

    print(f"PHASE 2A training complete")
    print(f"LightGBM version: {lgb.__version__}")
    print(f"CPU architecture: {platform.machine()}")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Best iteration: {model.best_iteration_}")
    print(f"Top 10 feature: {feature_importance.head(10)['feature'].tolist()}")


if __name__ == "__main__":
    main()
