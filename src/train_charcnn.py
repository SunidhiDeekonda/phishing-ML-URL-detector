from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 8
PATIENCE = 2
MIN_IMPROVEMENT = 1e-4
EMBEDDING_DIM = 16
KERNEL_SIZES = [3, 5, 7]
FILTERS_PER_BRANCH = 128
DENSE_DIM = 64
DROPOUT = 0.3
VOCAB_SIZE = 51
MAX_SEQUENCE_LENGTH = 200


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_roc_auc: float
    val_pr_auc: float


class UrlCharDataset(Dataset):
    """Character sequence dataset preserving row alignment."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray, row_indices: np.ndarray):
        self.sequences = sequences.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.row_indices = row_indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.as_tensor(self.sequences[idx], dtype=torch.long),
            torch.as_tensor(self.labels[idx], dtype=torch.float32),
            int(self.row_indices[idx]),
        )


class CharCNN(nn.Module):
    """Three-branch character CNN used by the reference PhishX-style model."""

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
            nn.Linear(in_features=FILTERS_PER_BRANCH * len(KERNEL_SIZES), out_features=DENSE_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(in_features=DENSE_DIM, out_features=1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, 200)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = UrlCharDataset(
        sequences=sequences[train_idx],
        labels=labels[train_idx],
        row_indices=train_idx,
    )
    val_dataset = UrlCharDataset(
        sequences=sequences[val_idx],
        labels=labels[val_idx],
        row_indices=val_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels: List[int] = []
    all_probs: List[float] = []

    for sequences, labels, _ in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(sequences)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    if total_samples == 0:
        raise ValueError("Loader produced no batches")

    avg_loss = total_loss / total_samples
    if not np.isfinite(avg_loss):
        raise ValueError("Training encountered NaN/Inf loss")

    return avg_loss, np.asarray(all_labels, dtype=int), np.asarray(all_probs, dtype=np.float64)


def evaluate_validation(
    model: nn.Module,
    sequences: np.ndarray,
    labels: np.ndarray,
    val_idx: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    model.to(device)

    val_dataset = UrlCharDataset(
        sequences=sequences[val_idx],
        labels=labels[val_idx],
        row_indices=val_idx,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.BCEWithLogitsLoss()

    val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, None, device)
    return val_loss, val_labels, val_probs, val_idx.astype(np.int64)


def train_charcnn(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    device_name: str,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
) -> Tuple[dict, List[EpochMetrics], int, nn.Module]:
    device = torch.device(device_name)

    train_loader, val_loader = build_dataloaders(sequences, labels, train_idx, val_idx, batch_size)
    model = CharCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_roc = -float("inf")
    best_epoch = -1
    no_improve_epochs = 0
    best_state = None
    history: List[EpochMetrics] = []

    for epoch in range(1, max_epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, None, device)
        metrics = binary_metrics(val_labels, val_probs, threshold=0.5)

        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            val_accuracy=float(metrics["accuracy"]),
            val_precision=float(metrics["precision"]),
            val_recall=float(metrics["recall"]),
            val_f1=float(metrics["f1"]),
            val_roc_auc=float(metrics["roc_auc"]),
            val_pr_auc=float(metrics["pr_auc"]),
        )
        history.append(epoch_metrics)

        print(f"Epoch {epoch}/{max_epochs}")
        print(f"train_loss: {epoch_metrics.train_loss:.6f}")
        print(f"val_loss: {epoch_metrics.val_loss:.6f}")
        print(f"val_accuracy: {epoch_metrics.val_accuracy:.6f}")
        print(f"val_precision: {epoch_metrics.val_precision:.6f}")
        print(f"val_recall: {epoch_metrics.val_recall:.6f}")
        print(f"val_f1: {epoch_metrics.val_f1:.6f}")
        print(f"val_roc_auc: {epoch_metrics.val_roc_auc:.6f}")
        print(f"val_pr_auc: {epoch_metrics.val_pr_auc:.6f}")

        if epoch_metrics.val_roc_auc > best_roc + MIN_IMPROVEMENT:
            best_roc = epoch_metrics.val_roc_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE}).")
                break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("No best epoch found; check training data/labels")

    model.load_state_dict(best_state)
    return {
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "early_stopping": best_epoch < max_epochs and no_improve_epochs >= PATIENCE,
    }, history, best_epoch, model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_with_fallback(
    sequences: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[dict, List[EpochMetrics], int, nn.Module, str, int, bool, List[str]]:
    requested_device = "MPS" if torch.backends.mps.is_available() else "CUDA" if torch.cuda.is_available() else "CPU"
    chosen_device = "CPU"
    warnings: List[str] = []

    if requested_device == "MPS" and torch.backends.mps.is_available():
        chosen_device = "MPS"
    elif requested_device == "CUDA" and torch.cuda.is_available():
        chosen_device = "CUDA"

    current_batch_size = BATCH_SIZE

    for attempt_device in [chosen_device, "CPU"]:
        # Do not retry batch size reduction on CPU because that does not materially
        # change memory footprint in this dataset and can mask device issues.
        for bs in [current_batch_size, 32]:
            if attempt_device == "CPU" and bs == 32:
                break
            try:
                stats, history, best_epoch, model = train_charcnn(
                    sequences=sequences,
                    labels=labels,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    device_name=attempt_device.lower(),
                    batch_size=bs,
                    max_epochs=MAX_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                return stats, history, best_epoch, model, attempt_device, bs, attempt_device != chosen_device, warnings
            except RuntimeError as exc:
                error_msg = str(exc).lower()
                if "out of memory" in error_msg and attempt_device == "MPS" and bs == BATCH_SIZE:
                    warnings.append("MPS OOM encountered; retrying with batch_size=32")
                    continue
                if "not supported" in error_msg and attempt_device == "MPS":
                    warnings.append("MPS operation unsupported; falling back to CPU")
                    break
                raise

        if attempt_device == "MPS" and ("MPS OOM encountered; retrying with batch_size=32" in warnings or
                                        "MPS operation unsupported; falling back to CPU" in warnings):
            continue
    raise RuntimeError("Training did not complete on available devices")


def main() -> None:
    set_seed(SEED)

    # Ensure deterministic behavior where supported by backend.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "data" / "processed" / "dataset.csv"
    seq_path = root / "data" / "processed" / "char_sequences.npy"
    train_idx_path = root / "data" / "processed" / "train_idx.npy"
    val_idx_path = root / "data" / "processed" / "val_idx.npy"
    test_idx_path = root / "data" / "processed" / "test_idx.npy"

    model_dir = root / "models"
    results_dir = root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "char_cnn.pt"
    history_path = results_dir / "charcnn_training_history.json"
    pred_path = results_dir / "charcnn_validation_predictions.csv"

    dataset = pd.read_csv(dataset_path)
    sequences = np.load(seq_path)
    labels = dataset["label"].to_numpy(dtype=np.int64)
    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    if sequences.shape != (20_000, 200):
        raise ValueError(f"Expected char_sequences shape (20000, 200), got {sequences.shape}")
    if np.max(sequences) > 50 or np.min(sequences) < 0:
        raise ValueError("Character indices must be in the range 0..50")
    if sequences.shape[0] != len(labels):
        raise ValueError("Sequence and label lengths do not match")

    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())
    if not (train_set.isdisjoint(val_set) and train_set.isdisjoint(test_set) and val_set.isdisjoint(test_set)):
        raise ValueError("Train/val/test indices overlap")

    train_start = time.perf_counter()
    stats, history, best_epoch, best_model, device_used, final_batch_size, used_fallback, fallback_warnings = train_with_fallback(
        sequences=sequences,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    training_duration = time.perf_counter() - train_start

    start_inference = time.perf_counter()
    device_for_infer = torch.device("mps" if device_used == "MPS" and torch.backends.mps.is_available() else
                                    "cuda" if device_used == "CUDA" and torch.cuda.is_available() else
                                    "cpu")
    val_loss, val_labels, val_probs, val_row_idx = evaluate_validation(
        best_model,
        sequences,
        labels,
        val_idx,
        device_for_infer,
        batch_size=final_batch_size,
    )
    _infer_time = time.perf_counter() - start_inference

    if len(val_probs) != len(val_idx):
        raise ValueError("Validation prediction rows mismatch")
    if np.any(~np.isfinite(val_probs)):
        raise ValueError("Validation probabilities contain NaN/Inf")
    if np.any((val_probs < 0) | (val_probs > 1)):
        raise ValueError("Validation probabilities outside [0, 1]")

    final_metrics = binary_metrics(val_labels, val_probs, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(val_labels, (val_probs >= 0.5).astype(int), labels=[0, 1]).ravel()
    if (tn + fp + fn + tp) != 2_000:
        raise ValueError("Confusion matrix does not sum to 2,000")

    # Re-serialize best checkpoint with metadata
    best_val_roc_auc = max(m.val_roc_auc for m in history) if history else float("nan")
    checkpoint = {
        "model_state_dict": {k: v.cpu() for k, v in best_model.state_dict().items()},
        "best_epoch": int(best_epoch),
        "best_validation_roc_auc": float(best_val_roc_auc),
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "random_seed": SEED,
        "architecture": {
            "embedding_dim": EMBEDDING_DIM,
            "conv_kernels": KERNEL_SIZES,
            "filters_per_branch": FILTERS_PER_BRANCH,
            "dense_dim": DENSE_DIM,
            "dropout": DROPOUT,
        },
    }
    torch.save(checkpoint, model_path)

    pd.DataFrame(
        {
            "row_index": val_row_idx,
            "true_label": val_labels,
            "cnn_probability": val_probs,
            "cnn_prediction": (val_probs >= 0.5).astype(int),
        }
    ).to_csv(pred_path, index=False)

    # Training history JSON used for reproducibility and reporting
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "CharCNN",
                "device_requested": "MPS" if torch.backends.mps.is_available() else "CUDA" if torch.cuda.is_available() else "CPU",
                "device_used": device_used,
                "batch_size": final_batch_size,
                "learning_rate": LEARNING_RATE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "minimum_improvement": MIN_IMPROVEMENT,
                "epochs_completed": int(stats["epochs_completed"]),
                "best_epoch": int(best_epoch),
                "early_stopping_triggered": bool(stats["early_stopping"]),
                "training_duration_seconds": float(training_duration),
                "history": [asdict(m) for m in history],
                "final_validation_metrics": final_metrics,
                "final_confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "final_validation_loss": float(val_loss),
                "inference_duration_seconds": float(_infer_time),
            },
            f,
            indent=2,
        )

    if not model_path.exists():
        raise FileNotFoundError("Character-CNN model file missing")
    if not history_path.exists():
        raise FileNotFoundError("History JSON file missing")
    if not pred_path.exists():
        raise FileNotFoundError("Validation predictions CSV file missing")

    print(f"Phase 2B training complete. Device requested: MPS, used: {device_used}")
    print(f"Training duration: {stats['epochs_completed']} epochs executed")
    print(f"Best epoch: {best_epoch}")
    if used_fallback:
        for warning in fallback_warnings:
            print(f"Fallback warning: {warning}")


if __name__ == "__main__":
    main()
