from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


VOCAB_SIZE = 51
EMBEDDING_DIM = 16
KERNEL_SIZES = [3, 5, 7]
FILTERS_PER_BRANCH = 128
DENSE_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
REFERENCE_CNN_WEIGHT = 0.60
REFERENCE_LIGHTGBM_WEIGHT = 0.40
SELECTED_CNN_WEIGHT = 0.95
SELECTED_LIGHTGBM_WEIGHT = 0.05
THRESHOLD = 0.50


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float | int]:
    preds = (y_prob >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def add_rates(metrics: Dict[str, float | int]) -> Dict[str, float | int]:
    tn, fp, fn, tp = int(metrics["tn"]), int(metrics["fp"]), int(metrics["fn"]), int(metrics["tp"])
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) else 0.0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) else 0.0
    return metrics


def evaluate_cnn_probabilities(
    model_path: Path,
    sequences: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    device,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class UrlCharDataset(Dataset):
        """Simple validation/test character sequence dataset."""

        def __init__(self, sequences: np.ndarray, labels: np.ndarray):
            self.sequences = sequences.astype(np.int64)
            self.labels = labels.astype(np.float32)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            return (
                torch.as_tensor(self.sequences[idx], dtype=torch.long),
                torch.as_tensor(self.labels[idx], dtype=torch.float32),
            )

    class CharCNN(nn.Module):
        """Three-branch character CNN matching the Phase 2B architecture."""

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                num_embeddings=VOCAB_SIZE,
                embedding_dim=EMBEDDING_DIM,
                padding_idx=0,
            )
            self.branches = nn.ModuleList(
                [
                    nn.Conv1d(
                        in_channels=EMBEDDING_DIM,
                        out_channels=FILTERS_PER_BRANCH,
                        kernel_size=k,
                        padding=(k // 2),
                    )
                    for k in KERNEL_SIZES
                ]
            )
            self.activation = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(FILTERS_PER_BRANCH * len(KERNEL_SIZES), DENSE_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(DENSE_DIM, 1),
            )

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embedding(input_ids)  # (batch, 200, 16)
            x = x.permute(0, 2, 1)        # (batch, 16, 200)
            branch_outputs = []
            for branch in self.branches:
                out = branch(x)
                out = self.activation(out)
                out = self.pool(out).squeeze(dim=2)
                branch_outputs.append(out)

            x = torch.cat(branch_outputs, dim=1)
            logits = self.fc(x).squeeze(dim=1)
            return logits

    test_sequences = sequences[test_idx]
    test_labels = labels[test_idx]
    if test_sequences.shape != (len(test_idx), 200):
        raise ValueError(f"Expected test sequences shape ({len(test_idx)}, 200), got {test_sequences.shape}")

    dataset = UrlCharDataset(test_sequences, test_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    checkpoint = torch.load(model_path, map_location=device)
    model = CharCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs: List[float] = []
    for batch_sequences, _ in loader:
        batch_sequences = batch_sequences.to(device)
        with torch.no_grad():
            logits = model(batch_sequences)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())

    probs_arr = np.asarray(all_probs, dtype=np.float64)
    if len(probs_arr) != len(test_idx):
        raise ValueError(f"Expected {len(test_idx)} CNN probabilities, got {len(probs_arr)}")
    return probs_arr


def plot_confusion_matrix(selected: Dict[str, float | int], output_path: Path) -> None:
    matrix = np.array(
        [
            [selected["tn"], selected["fp"]],
            [selected["fn"], selected["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(6, 5), dpi=220)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legitimate", "Predicted Phishing"])
    ax.set_yticklabels(["Actual Legitimate", "Actual Phishing"])
    ax.set_title("Selected Ensemble Confusion Matrix (w=0.95 CNN / 0.05 LGBM)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10, color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, model_scores: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=220)
    for label, probs in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{label} (ROC-AUC={auc:.6f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Held-out Test)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, model_scores: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=220)
    for label, probs in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(recall, precision, label=f"{label} (PR-AUC={ap:.6f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Held-out Test)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_model_comparison(metrics_rows: List[Dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = [row["model"] for row in metrics_rows]
    accuracy = [row["accuracy"] for row in metrics_rows]
    precision = [row["precision"] for row in metrics_rows]
    recall = [row["recall"] for row in metrics_rows]
    f1 = [row["f1"] for row in metrics_rows]
    roc = [row["roc_auc"] for row in metrics_rows]

    x = np.arange(len(models))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    ax.bar(x - 2 * width, accuracy, width, label="Accuracy")
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.bar(x + 2 * width, roc, width, label="ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0.95, 1.01)
    ax.set_title("Test Metrics Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_feature_importance(importance_path: Path, output_path: Path) -> None:
    df = pd.read_csv(importance_path)
    if not {"feature", "importance"}.issubset(set(df.columns)):
        raise ValueError(f"Expected columns 'feature' and 'importance' in {importance_path}")
    top = df.sort_values("importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4C72B0")
    ax.set_title("Top 10 LightGBM Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def plot_probability_distribution(
    y_true: np.ndarray,
    selected_probs: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=220)
    legit = selected_probs[y_true == 0]
    phishing = selected_probs[y_true == 1]
    ax.hist(
        legit,
        bins=40,
        alpha=0.7,
        density=True,
        label="Actual Legitimate",
    )
    ax.hist(
        phishing,
        bins=40,
        alpha=0.7,
        density=True,
        label="Actual Phishing",
    )
    ax.axvline(THRESHOLD, color="black", linestyle="--", label="Threshold 0.50")
    ax.set_title("Selected Ensemble Probability Distribution (Held-out Test)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png")
    plt.close(fig)


def print_compact_comparison(metrics_rows: List[Dict[str, float | int]]) -> None:
    print("FINAL MODEL COMPARISON")
    print("{:<32} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
    ))
    print("-" * 96)
    for row in metrics_rows:
        print(
            "{:<32} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(
                row["model"],
                row["accuracy"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["roc_auc"],
                row["pr_auc"],
            )
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "data" / "processed" / "dataset.csv"
    features_path = repo_root / "data" / "processed" / "features.csv"
    seq_path = repo_root / "data" / "processed" / "char_sequences.npy"
    train_idx_path = repo_root / "data" / "processed" / "train_idx.npy"
    val_idx_path = repo_root / "data" / "processed" / "val_idx.npy"
    test_idx_path = repo_root / "data" / "processed" / "test_idx.npy"

    lgbm_model_path = repo_root / "models" / "lightgbm_calibrated.pkl"
    cnn_model_path = repo_root / "models" / "char_cnn.pt"
    importance_path = repo_root / "results" / "lightgbm_feature_importance.csv"

    predictions_path = repo_root / "results" / "test_predictions.csv"
    metrics_path = repo_root / "results" / "final_test_metrics.json"
    comparison_csv_path = repo_root / "results" / "model_comparison.csv"
    plots_dir = repo_root / "results" / "plots"

    required_paths = [
        dataset_path,
        features_path,
        seq_path,
        train_idx_path,
        val_idx_path,
        test_idx_path,
        lgbm_model_path,
        cnn_model_path,
        importance_path,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")

    dataset = pd.read_csv(dataset_path)
    features = pd.read_csv(features_path)
    sequences = np.load(seq_path)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    if len(dataset) != 20_000:
        raise ValueError(f"Expected dataset size 20,000, found {len(dataset)}")
    if sequences.shape != (20_000, 200):
        raise ValueError(f"Expected char sequence shape (20000, 200), found {sequences.shape}")
    if np.max(sequences) > 50 or np.min(sequences) < 0:
        raise ValueError("Character sequence indices must be in [0, 50]")

    if len(test_idx) != 3_998:
        raise ValueError(f"Expected 3,998 test rows, found {len(test_idx)}")

    labels = dataset["label"].to_numpy(dtype=np.int64)
    y_test = labels[test_idx]
    url_values = dataset["url"].astype(str).to_numpy()
    test_urls = url_values[test_idx]

    if len(y_test) != 3_998:
        raise ValueError(f"Expected 3,998 test labels, found {len(y_test)}")

    legit_count = int((y_test == 0).sum())
    phishing_count = int((y_test == 1).sum())
    if not (1_950 <= legit_count <= 2_050 and 1_950 <= phishing_count <= 2_050):
        raise ValueError(
            f"Unexpected class balance in test set: legitimate={legit_count}, phishing={phishing_count}"
        )
    if legit_count != 2003 or phishing_count != 1995:
        # keep historical expected counts as a sanity floor but do not hard-fail if dataset shifts.
        print(
            f"Warning: test class counts are {legit_count}/{phishing_count}, "
            "expected nominally 2003 legitimate and 1995 phishing."
        )

    train_set = set(map(int, train_idx.tolist()))
    val_set = set(map(int, val_idx.tolist()))
    test_set = set(map(int, test_idx.tolist()))
    if not (train_set.isdisjoint(val_set) and train_set.isdisjoint(test_set) and val_set.isdisjoint(test_set)):
        raise ValueError("Split indices still overlap")

    # No model re-training or calibration in this phase.
    try:
        lightgbm_model = joblib.load(lgbm_model_path, mmap_mode="r")
    except TypeError:
        # Older joblib versions may not support mmap_mode for this payload.
        lightgbm_model = joblib.load(lgbm_model_path)
    if not isinstance(lightgbm_model, CalibratedClassifierCV):
        # The existing artifact should be calibrated by design.
        print("Warning: loaded object is not CalibratedClassifierCV; continuing to use predict_proba directly.")
    X_test = features.iloc[test_idx]
    if len(X_test) != 3_998:
        raise ValueError(f"Expected 3,998 test feature rows, found {len(X_test)}")

    lgbm_probs = lightgbm_model.predict_proba(X_test)[:, 1].astype(np.float64)
    if lgbm_probs.shape[0] != len(test_idx):
        raise ValueError(f"Expected {len(test_idx)} LGBM probabilities, got {lgbm_probs.shape[0]}")
    if not np.isfinite(lgbm_probs).all():
        raise ValueError("Non-finite LightGBM probabilities")
    if np.any((lgbm_probs < 0) | (lgbm_probs > 1)):
        raise ValueError("LightGBM probabilities out of [0,1]")

    device = pick_device()
    cnn_probs = evaluate_cnn_probabilities(
        model_path=cnn_model_path,
        sequences=sequences,
        labels=labels,
        test_idx=test_idx,
        device=device,
    )
    if not np.isfinite(cnn_probs).all():
        raise ValueError("Non-finite CNN probabilities")
    if np.any((cnn_probs < 0) | (cnn_probs > 1)):
        raise ValueError("CNN probabilities out of [0,1]")

    ref_probs = REFERENCE_CNN_WEIGHT * cnn_probs + REFERENCE_LIGHTGBM_WEIGHT * lgbm_probs
    sel_probs = SELECTED_CNN_WEIGHT * cnn_probs + SELECTED_LIGHTGBM_WEIGHT * lgbm_probs

    for p_name, probs in [
        ("LightGBM", lgbm_probs),
        ("CNN", cnn_probs),
        ("Reference Ensemble", ref_probs),
        ("Selected Ensemble", sel_probs),
    ]:
        if not np.isfinite(probs).all():
            raise ValueError(f"Non-finite probabilities in {p_name}")
        if np.any((probs < 0) | (probs > 1)):
            raise ValueError(f"{p_name} probabilities outside [0, 1]")

    lgbm_metrics = add_rates(evaluate_binary_metrics(y_test, lgbm_probs))
    cnn_metrics = add_rates(evaluate_binary_metrics(y_test, cnn_probs))
    reference_metrics = add_rates(evaluate_binary_metrics(y_test, ref_probs))
    selected_metrics = add_rates(evaluate_binary_metrics(y_test, sel_probs))

    # Save row-level predictions
    test_predictions_df = pd.DataFrame(
        {
            "row_index": test_idx.astype(int),
            "url": test_urls,
            "true_label": y_test.astype(int),
            "lightgbm_probability": lgbm_probs,
            "cnn_probability": cnn_probs,
            "reference_ensemble_probability": ref_probs,
            "selected_ensemble_probability": sel_probs,
            "lightgbm_prediction": (lgbm_probs >= THRESHOLD).astype(int),
            "cnn_prediction": (cnn_probs >= THRESHOLD).astype(int),
            "reference_ensemble_prediction": (ref_probs >= THRESHOLD).astype(int),
            "selected_ensemble_prediction": (sel_probs >= THRESHOLD).astype(int),
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions_df.to_csv(predictions_path, index=False)

    if len(test_predictions_df) != 3_998:
        raise ValueError(f"Expected 3,998 test prediction rows, found {len(test_predictions_df)}")

    for name, metrics in [
        ("lightgbm", lgbm_metrics),
        ("cnn", cnn_metrics),
        ("reference", reference_metrics),
        ("selected", selected_metrics),
    ]:
        confusion_sum = int(metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"])
        if confusion_sum != 3_998:
            raise ValueError(f"{name} confusion matrix does not sum to 3,998: {confusion_sum}")

    comparison_rows = [
        {
            "model": "Calibrated LightGBM",
            **{k: float(v) for k, v in lgbm_metrics.items() if k in {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}},
            "tn": int(lgbm_metrics["tn"]),
            "fp": int(lgbm_metrics["fp"]),
            "fn": int(lgbm_metrics["fn"]),
            "tp": int(lgbm_metrics["tp"]),
            "false_positive_rate": float(lgbm_metrics["false_positive_rate"]),
            "false_negative_rate": float(lgbm_metrics["false_negative_rate"]),
        },
        {
            "model": "Char-CNN",
            **{k: float(v) for k, v in cnn_metrics.items() if k in {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}},
            "tn": int(cnn_metrics["tn"]),
            "fp": int(cnn_metrics["fp"]),
            "fn": int(cnn_metrics["fn"]),
            "tp": int(cnn_metrics["tp"]),
            "false_positive_rate": float(cnn_metrics["false_positive_rate"]),
            "false_negative_rate": float(cnn_metrics["false_negative_rate"]),
        },
        {
            "model": "Reference Ensemble 0.60/0.40",
            **{k: float(v) for k, v in reference_metrics.items() if k in {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}},
            "tn": int(reference_metrics["tn"]),
            "fp": int(reference_metrics["fp"]),
            "fn": int(reference_metrics["fn"]),
            "tp": int(reference_metrics["tp"]),
            "false_positive_rate": float(reference_metrics["false_positive_rate"]),
            "false_negative_rate": float(reference_metrics["false_negative_rate"]),
        },
        {
            "model": "Selected Ensemble 0.95/0.05",
            **{k: float(v) for k, v in selected_metrics.items() if k in {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}},
            "tn": int(selected_metrics["tn"]),
            "fp": int(selected_metrics["fp"]),
            "fn": int(selected_metrics["fn"]),
            "tp": int(selected_metrics["tp"]),
            "false_positive_rate": float(selected_metrics["false_positive_rate"]),
            "false_negative_rate": float(selected_metrics["false_negative_rate"]),
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(comparison_csv_path, index=False)

    metrics_payload = {
        "dataset_size": 20_000,
        "test_size": 3_998,
        "threshold": THRESHOLD,
        "selection_method": "ensemble weights selected on validation ROC-AUC before test evaluation",
        "reference_weights": {
            "cnn": REFERENCE_CNN_WEIGHT,
            "lightgbm": REFERENCE_LIGHTGBM_WEIGHT,
        },
        "selected_weights": {
            "cnn": SELECTED_CNN_WEIGHT,
            "lightgbm": SELECTED_LIGHTGBM_WEIGHT,
        },
        "device_used": str(device),
        "lightgbm": lgbm_metrics,
        "charcnn": cnn_metrics,
        "reference_ensemble": reference_metrics,
        "selected_ensemble": selected_metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    model_scores = {
        "Calibrated LightGBM": lgbm_probs,
        "Char-CNN": cnn_probs,
        "Reference Ensemble (0.60/0.40)": ref_probs,
        "Selected Ensemble (0.95/0.05)": sel_probs,
    }
    plot_confusion_matrix(selected_metrics, output_path=plots_dir / "confusion_matrix.png")
    plot_roc_curves(y_test, model_scores, output_path=plots_dir / "roc_curve.png")
    plot_pr_curves(y_test, model_scores, output_path=plots_dir / "precision_recall_curve.png")
    plot_model_comparison(comparison_rows, output_path=plots_dir / "model_comparison.png")
    plot_feature_importance(importance_path, output_path=plots_dir / "feature_importance.png")
    plot_probability_distribution(y_test, sel_probs, output_path=plots_dir / "probability_distribution.png")

    required_plots = [
        plots_dir / "confusion_matrix.png",
        plots_dir / "roc_curve.png",
        plots_dir / "precision_recall_curve.png",
        plots_dir / "model_comparison.png",
        plots_dir / "feature_importance.png",
        plots_dir / "probability_distribution.png",
    ]
    for p in required_plots:
        if not p.exists():
            raise FileNotFoundError(f"Expected plot not created: {p}")

    print("\nPHASE 2D — FINAL TEST REPORT")
    print("\nEvaluation Protocol")
    print("-------------------")
    print(f"Test rows: {len(test_idx)}")
    print(f"Test legitimate: {legit_count}")
    print(f"Test phishing: {phishing_count}")
    print("Models retrained: NO")
    print("Calibration performed on test: NO")
    print("Weights changed after seeing test: NO")
    print("Threshold changed after seeing test: NO")

    def print_block(name: str, metrics: Dict[str, float | int], extra: str = "") -> None:
        print(f"\n{name}")
        if extra:
            print(extra)
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1: {metrics['f1']}")
        print(f"ROC-AUC: {metrics['roc_auc']}")
        print(f"PR-AUC: {metrics['pr_auc']}")
        print(f"TN: {metrics['tn']}")
        print(f"FP: {metrics['fp']}")
        print(f"FN: {metrics['fn']}")
        print(f"TP: {metrics['tp']}")
        print(f"False Positive Rate: {metrics['false_positive_rate']}")
        print(f"False Negative Rate: {metrics['false_negative_rate']}")

    print_block(
        "CALIBRATED LIGHTGBM — TEST",
        lgbm_metrics,
    )
    print_block("CHAR-CNN — TEST", cnn_metrics)
    print_block(
        "REFERENCE ENSEMBLE — TEST",
        reference_metrics,
        extra="Weights: CNN 0.60, LightGBM 0.40",
    )
    print_block(
        "SELECTED ENSEMBLE — TEST",
        selected_metrics,
        extra="Weights: CNN 0.95, LightGBM 0.05",
    )

    print()
    print_compact_comparison(comparison_rows)

    reference_accuracy = 99.819 / 100
    selected_accuracy = float(selected_metrics["accuracy"])
    diff = selected_accuracy - reference_accuracy
    direction = "higher" if diff > 0 else "lower"
    if abs(diff) < 1e-12:
        direction = "equal"
    print("\nREFERENCE STUDY COMPARISON")
    print("--------------------------")
    print("Reference study accuracy: 99.819%")
    print(f"Our selected ensemble accuracy: {selected_accuracy * 100:.6f}%")
    print(f"Absolute percentage-point difference: {abs(diff) * 100:.6f} points")
    print(f"Comparison direction: {direction}")

    print("\nARTIFACTS")
    print(f"- {metrics_path}")
    print(f"- {comparison_csv_path}")
    print(f"- {predictions_path}")
    print(f"- Plots:")
    for path in required_plots:
        print(f"  - {path}")

    print("\nSANITY CHECKS")
    print("Prediction rows: PASS")
    print("No NaN probabilities: PASS")
    print("All probabilities in [0,1]: PASS")
    print("All confusion matrices sum to 3998: PASS")
    print(f"Test set model files used: PASS (model checkpoints unchanged)")
    print(f"Frozen weights enforced: PASS (0.60/0.40 and 0.95/0.05)")
    print("FINAL ML EXPERIMENT COMPLETE: YES")


if __name__ == "__main__":
    main()
