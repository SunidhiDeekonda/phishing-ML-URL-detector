from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float | int]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def format_metric_line(name: str, metrics: Dict[str, float | int]) -> str:
    return (
        f"{name} - Accuracy: {metrics['accuracy']:.6f}, "
        f"Precision: {metrics['precision']:.6f}, Recall: {metrics['recall']:.6f}, "
        f"F1: {metrics['f1']:.6f}, ROC-AUC: {metrics['roc_auc']:.6f}, "
        f"PR-AUC: {metrics['pr_auc']:.6f}, TN: {metrics['tn']}, FP: {metrics['fp']}, "
        f"FN: {metrics['fn']}, TP: {metrics['tp']}"
    )


def select_best_weight(candidates: List[Dict[str, float]], tie_tolerance: float = 1e-5) -> Dict[str, float]:
    if not candidates:
        raise ValueError("No candidates provided for ensemble weight selection")

    best_roc = max(candidate["roc_auc"] for candidate in candidates)
    best_candidates = [
        candidate for candidate in candidates if abs(candidate["roc_auc"] - best_roc) <= tie_tolerance
    ]
    if len(best_candidates) == 1:
        return best_candidates[0]

    # Secondary preference: closest weight to the reference CNN weight 0.60.
    return min(
        best_candidates,
        key=lambda candidate: abs(candidate["cnn_weight"] - 0.60),
    )


def weight_grid(step: float = 0.05) -> Iterable[float]:
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        yield round(i * step, 10)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    lgbm_pred_path = repo_root / "results" / "lightgbm_validation_predictions.csv"
    cnn_pred_path = repo_root / "results" / "charcnn_validation_predictions.csv"
    weight_search_path = repo_root / "results" / "ensemble_weight_search.csv"
    config_path = repo_root / "results" / "ensemble_config.json"

    if not lgbm_pred_path.exists():
        raise FileNotFoundError(f"Missing file: {lgbm_pred_path}")
    if not cnn_pred_path.exists():
        raise FileNotFoundError(f"Missing file: {cnn_pred_path}")

    lgbm_df = pd.read_csv(lgbm_pred_path)
    cnn_df = pd.read_csv(cnn_pred_path)

    # Validate required columns exist.
    lgbm_required = {"row_index", "true_label", "calibrated_probability"}
    cnn_required = {"row_index", "true_label", "cnn_probability"}
    missing_lgbm = sorted(lgbm_required - set(lgbm_df.columns))
    missing_cnn = sorted(cnn_required - set(cnn_df.columns))
    if missing_lgbm:
        raise ValueError(f"lightgbm_validation_predictions.csv is missing columns: {missing_lgbm}")
    if missing_cnn:
        raise ValueError(f"charcnn_validation_predictions.csv is missing columns: {missing_cnn}")

    # Ensure row_index values are unique and valid.
    if lgbm_df["row_index"].duplicated().any():
        duplicates = lgbm_df[lgbm_df["row_index"].duplicated()]["row_index"].tolist()
        raise ValueError(f"Duplicate row_index values in LightGBM predictions: {duplicates[:10]}")
    if cnn_df["row_index"].duplicated().any():
        duplicates = cnn_df[cnn_df["row_index"].duplicated()]["row_index"].tolist()
        raise ValueError(f"Duplicate row_index values in CNN predictions: {duplicates[:10]}")

    merged = pd.merge(
        lgbm_df[["row_index", "true_label", "calibrated_probability"]],
        cnn_df[["row_index", "true_label", "cnn_probability"]],
        on="row_index",
        how="inner",
        suffixes=("_lgbm", "_cnn"),
    )

    if len(merged) != 2_000:
        raise ValueError(
            f"Expected exactly 2,000 matched validation rows after alignment, found {len(merged)}"
        )
    if merged["row_index"].duplicated().any():
        raise ValueError("Merged validation rows contain duplicate row_index values")

    if not np.array_equal(merged["true_label_lgbm"].to_numpy(), merged["true_label_cnn"].to_numpy()):
        mismatch = np.nonzero(merged["true_label_lgbm"] != merged["true_label_cnn"])[0][:10].tolist()
        raise ValueError(f"True label mismatch between prediction files at positions: {mismatch}")

    y_true = merged["true_label_lgbm"].astype(int).to_numpy()

    lgbm_probs = merged["calibrated_probability"].to_numpy(dtype=float)
    cnn_probs = merged["cnn_probability"].to_numpy(dtype=float)

    # General validation checks
    if len(y_true) != 2_000:
        raise ValueError(f"Expected exactly 2,000 labels, found {len(y_true)}")
    if np.isnan(y_true).any():
        raise ValueError("Found missing true labels")

    for label_name, probs in [("calibrated LGBM", lgbm_probs), ("CNN", cnn_probs)]:
        if np.isnan(probs).any() or np.any((probs < 0) | (probs > 1)):
            raise ValueError(f"Invalid probability values in {label_name} predictions")

    # Base model metrics
    lgbm_metrics = evaluate_binary_metrics(y_true, lgbm_probs)
    cnn_metrics = evaluate_binary_metrics(y_true, cnn_probs)

    reference_weight = 0.60
    reference_probs = reference_weight * cnn_probs + (1.0 - reference_weight) * lgbm_probs
    reference_metrics = evaluate_binary_metrics(y_true, reference_probs)

    weight_rows: List[Dict[str, float | int]] = []
    for cnn_weight in weight_grid(0.05):
        lightgbm_weight = 1.0 - cnn_weight
        ensemble_probs = cnn_weight * cnn_probs + lightgbm_weight * lgbm_probs
        metrics = evaluate_binary_metrics(y_true, ensemble_probs)
        weight_rows.append(
            {
                "cnn_weight": float(cnn_weight),
                "lightgbm_weight": float(lightgbm_weight),
                "accuracy": float(metrics["accuracy"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "roc_auc": float(metrics["roc_auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "tp": int(metrics["tp"]),
            }
        )

    search_df = pd.DataFrame(weight_rows)
    if search_df.empty:
        raise RuntimeError("Weight search produced no candidates")

    selected = select_best_weight(weight_rows)
    best_metrics = {
        "accuracy": selected["accuracy"],
        "precision": selected["precision"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "roc_auc": selected["roc_auc"],
        "pr_auc": selected["pr_auc"],
        "tn": selected["tn"],
        "fp": selected["fp"],
        "fn": selected["fn"],
        "tp": selected["tp"],
    }

    search_df.to_csv(weight_search_path, index=False)

    config_payload = {
        "reference_weights": {
            "cnn": float(reference_weight),
            "lightgbm": float(1.0 - reference_weight),
        },
        "reference_ensemble_metrics": reference_metrics,
        "best_weights": {
            "cnn": float(selected["cnn_weight"]),
            "lightgbm": float(selected["lightgbm_weight"]),
            "roc_auc": float(selected["roc_auc"]),
            "selected": True,
        },
        "best_ensemble_metrics": best_metrics,
        "comparison": {
            "calibrated_lightgbm": lgbm_metrics,
            "char_cnn": cnn_metrics,
            "reference_ensemble": reference_metrics,
            "best_ensemble": best_metrics,
        },
        "selection_criterion": "roc_auc",
        "roc_auc_tie_tolerance": 1e-5,
        "row_count": int(len(merged)),
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print("PHASE 2C ENSEMBLE SELECTION")
    print("Validation rows used:", len(merged))
    print(format_metric_line("Calibrated LightGBM", lgbm_metrics))
    print(format_metric_line("Char-CNN", cnn_metrics))
    print(format_metric_line("Reference Ensemble (0.60 CNN / 0.40 LGBM)", reference_metrics))
    print(
        "Best Ensemble Weight by Validation ROC-AUC: "
        f"cnn={selected['cnn_weight']:.2f}, "
        f"lightgbm={selected['lightgbm_weight']:.2f}"
    )
    print(format_metric_line("Best Ensemble", best_metrics))
    print(f"Weight search CSV: {weight_search_path}")
    print(f"Ensemble config JSON: {config_path}")


if __name__ == "__main__":
    main()
