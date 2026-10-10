"""Train, validate, and export the CharCNN + BiLSTM + Attention Phishing URL detector.

Evaluates on:
1. Balanced Held-Out Test Set (6,000 URLs)
2. Curated Hard-Negative Test URLs (including github.com/login, wikipedia.org, etc.)
Exports directly to models/charcnn_bilstm.onnx with verified numerical parity.
"""

from __future__ import annotations

import onnx
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from src.charcnn_bilstm import (
    CharCNNBiLSTMAttention,
    MAX_SEQUENCE_LENGTH,
    build_enhanced_vocab,
    encode_url_bilstm,
    encode_urls_bilstm,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data/processed"

SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 7
PATIENCE = 3


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UrlDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.as_tensor(sequences, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, dict, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for seqs, labels in loader:
            logits = model(seqs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)
    metrics = compute_metrics(y_true, y_prob)
    return avg_loss, metrics, y_prob


def main() -> None:
    seed_everything(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading Balanced Dataset & Hard Cases ===")
    dataset_path = DATA_DIR / "balanced_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing {dataset_path}. Run scripts/prepare_balanced_dataset.py first.")

    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} total rows from {dataset_path.name}.")

    hard_cases_path = ROOT / "tests/data/hard_legitimate_urls.json"
    hard_cases = json.loads(hard_cases_path.read_text(encoding="utf-8"))
    hard_dict = {c["url"]: c["hard_negative_split"] for c in hard_cases}

    # Partition: Assign hard cases to designated splits, then stratify the rest
    train_urls, train_labels = [], []
    val_urls, val_labels = [], []
    test_urls, test_labels = [], []

    remaining_df = []
    for _, row in df.iterrows():
        u = row["url"]
        l = int(row["label"])
        if u in hard_dict:
            split = hard_dict[u]
            if split == "train":
                for _ in range(8):
                    train_urls.append(u); train_labels.append(l)
            elif split == "validation":
                val_urls.append(u); val_labels.append(l)
            else:
                test_urls.append(u); test_labels.append(l)
        else:
            remaining_df.append(row)

    rem_df = pd.DataFrame(remaining_df)
    # Stratified split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    rem_train, rem_temp = train_test_split(rem_df, test_size=0.30, random_state=SEED, stratify=rem_df["label"])
    rem_val, rem_test = train_test_split(rem_temp, test_size=0.50, random_state=SEED, stratify=rem_temp["label"])

    train_urls.extend(rem_train["url"].tolist())
    train_labels.extend(rem_train["label"].tolist())

    val_urls.extend(rem_val["url"].tolist())
    val_labels.extend(rem_val["label"].tolist())

    test_urls.extend(rem_test["url"].tolist())
    test_labels.extend(rem_test["label"].tolist())

    print(f"Split sizes: Train={len(train_urls)}, Val={len(val_urls)}, Test={len(test_urls)}")

    print("=== Step 2: Tokenizing Sequences ===")
    vocab = build_enhanced_vocab()
    vocab_path = MODELS_DIR / "charcnn_bilstm_vocab.json"
    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"Saved vocabulary to {vocab_path} (size={len(vocab)}).")

    X_train = encode_urls_bilstm(train_urls, vocab, max_len=MAX_SEQUENCE_LENGTH)
    y_train = np.array(train_labels, dtype=np.float32)

    X_val = encode_urls_bilstm(val_urls, vocab, max_len=MAX_SEQUENCE_LENGTH)
    y_val = np.array(val_labels, dtype=np.float32)

    X_test = encode_urls_bilstm(test_urls, vocab, max_len=MAX_SEQUENCE_LENGTH)
    y_test = np.array(test_labels, dtype=np.float32)

    train_loader = DataLoader(UrlDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(UrlDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(UrlDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    print("=== Step 3: Initializing CharCNNBiLSTMAttention Model ===")
    model = CharCNNBiLSTMAttention(vocab_size=len(vocab))
    device = torch.device("cpu")
    model.to(device)

    # Class-weighted loss or label smoothed BCE
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_val_auc = 0.0
    best_weights_path = MODELS_DIR / "charcnn_bilstm_best.pt"
    patience_counter = 0

    print(f"=== Step 4: Training for up to {MAX_EPOCHS} Epochs on {device} ===")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(labels)

        avg_train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_metrics["roc_auc"])

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} [{elapsed:.1f}s] "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | Val FPR: {val_metrics['fpr']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), best_weights_path)
            patience_counter = 0
            print(f"  --> Saved new best model checkpoint (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds.")

    print("=== Step 5: Held-Out Test Evaluation ===")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    test_loss, test_metrics, _ = evaluate(model, test_loader, criterion)
    print("\n--- Held-Out Test Metrics ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Step 6: Hard-Negative Benchmark Verification ===")
    model.eval()
    hard_results = []
    with torch.no_grad():
        for case in hard_cases:
            encoded = encode_url_bilstm(case["url"], vocab, max_len=MAX_SEQUENCE_LENGTH)
            tensor = torch.as_tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
            prob = float(torch.sigmoid(model(tensor)).item())
            verdict = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
            status = "PASS" if verdict == case["expected_label"] else "FAIL"
            hard_results.append({
                "url": case["url"],
                "split": case["hard_negative_split"],
                "category": case["category"],
                "expected": case["expected_label"],
                "probability": round(prob, 4),
                "verdict": verdict,
                "status": status,
            })

    hard_df = pd.DataFrame(hard_results)
    print("\nHard Negatives by Split:")
    print(hard_df.groupby(["split", "status"]).size().unstack(fill_value=0))

    failures = hard_df[hard_df["status"] == "FAIL"]
    print(f"\nTotal Hard Negative Failures: {len(failures)} / {len(hard_df)}")
    if len(failures) > 0:
        print(failures[["url", "probability", "status"]].to_string())

    print("\n=== Step 7: Exporting to ONNX ===")
    onnx_path = MODELS_DIR / "charcnn_bilstm.onnx"
    dummy_input = torch.zeros((1, MAX_SEQUENCE_LENGTH), dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"Exported ONNX model to {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.2f} MB).")

    # Parity check
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample_seq = X_test[:10]
    with torch.no_grad():
        py_logits = model(torch.as_tensor(sample_seq, dtype=torch.long)).numpy()
    onnx_logits = sess.run(["logits"], {"input": sample_seq})[0]
    np.testing.assert_allclose(py_logits, onnx_logits, rtol=1e-3, atol=1e-3)
    print("Numerical parity test PASSED! PyTorch and ONNX match within tolerance.")

    # Save summary report
    summary = {
        "architecture": "CharCNN + BiLSTM + Attention",
        "parameters": sum(p.numel() for p in model.parameters()),
        "test_metrics": test_metrics,
        "hard_negatives": {
            "total": len(hard_df),
            "passed": int((hard_df["status"] == "PASS").sum()),
            "failed": int((hard_df["status"] == "FAIL").sum()),
            "pass_rate": float((hard_df["status"] == "PASS").mean()),
        },
    }
    metrics_path = RESULTS_DIR / "charcnn_bilstm_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation metrics to {metrics_path}.")


if __name__ == "__main__":
    main()
