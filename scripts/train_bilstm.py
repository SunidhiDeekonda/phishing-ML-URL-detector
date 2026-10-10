"""Train, evaluate, and export the CharCNN + BiLSTM + Attention URL detector.

Data: data/processed/balanced_dataset.csv (built by scripts/prepare_balanced_dataset.py).

Pipeline guarantees:
* every URL is cleaned (src/dataset_cleaning.py) and encoded with the same
  ``canonicalize_url`` the API uses, so training and serving inputs match;
* duplicates are removed on the canonical URL, so no URL appears in two splits;
* the split is grouped by registered domain (private PSL suffixes included);
* hard-negative URLs keep their designated split from tests/data/hard_legitimate_urls.json;
* any URL that is in the previous production model's held-out or adversarial
  test set is forced into the test split, so both models can be compared on
  those sets without leakage;
* the checkpoint is selected on validation ROC-AUC only; the test split is
  scored once, after selection.

Outputs: models/charcnn_bilstm.onnx, models/charcnn_bilstm_best.pt,
models/charcnn_bilstm_vocab.json, models/charcnn_bilstm_config.json,
results/charcnn_bilstm_metrics.json, results/charcnn_bilstm_test_set.csv.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import tldextract
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.charcnn_bilstm import MAX_SEQUENCE_LENGTH, build_enhanced_vocab, encode_urls_bilstm
from src.charcnn_bilstm_model import CharCNNBiLSTMAttention
from src.dataset_cleaning import clean_training_url
from src.url_normalization import canonicalize_url

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATASET_PATH = ROOT / "data/processed/balanced_dataset.csv"
MODEL_VERSION = "charcnn-bilstm-attention-v3.1"

SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_EPOCHS = 7
PATIENCE = 3
HARD_TRAIN_REPEAT = 8
THRESHOLD = 0.5
N_FOLDS = 20
TEST_FOLDS = {0, 1, 2}
VAL_FOLDS = {3, 4, 5}

EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def canonical_key(url: str) -> str:
    text = canonicalize_url(url).lower()
    text = text.split("://", 1)[-1]
    return text[4:] if text.startswith("www.") else text


def registered_domain(url: str) -> str:
    parts = EXTRACT(canonicalize_url(url))
    return ".".join(p for p in (parts.domain, parts.suffix) if p).lower() or canonicalize_url(url)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else None,
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def predict(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(sequences), 512):
            batch = torch.as_tensor(sequences[start:start + 512], dtype=torch.long)
            out.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(out)


def prepare_splits() -> tuple[dict[str, pd.DataFrame], dict]:
    raw = pd.read_csv(DATASET_PATH)
    report: dict[str, object] = {"source_rows": int(len(raw)), "source_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()}

    cleaned = raw.url.map(clean_training_url)
    raw["clean_url"] = cleaned.str[0]
    raw["cleaning"] = cleaned.str[1]
    report["cleaning"] = {
        f"{reason}|label={label}": int(count)
        for (reason, label), count in raw.groupby(["cleaning", "label"]).size().items()
    }
    data = raw[raw.clean_url.notna()].copy()
    data["key"] = data.clean_url.map(canonical_key)
    label_counts = data.groupby("key").label.nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    report["dropped_conflicting_label_keys"] = len(conflicting)
    data = data[~data.key.isin(conflicting)]
    report["dropped_duplicate_rows"] = int(data.key.duplicated().sum())
    data = data.drop_duplicates("key").reset_index(drop=True)
    data["group"] = data.clean_url.map(registered_domain)

    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text(encoding="utf-8"))
    hard_split = {canonical_key(c["url"]): c["hard_negative_split"] for c in hard}

    previous = pd.read_csv(ROOT / "data/processed/dataset.csv")
    previous_test = previous.iloc[np.load(ROOT / "data/processed/test_idx.npy")].url
    adversarial = pd.read_csv(ROOT / "data/processed/adversarial_test.csv")
    forced_test = {canonical_key(u) for u in pd.concat([previous_test, adversarial.original_url, adversarial.mutated_url]).astype(str)}

    data["split"] = ""
    data.loc[data.key.isin(hard_split), "split"] = data.loc[data.key.isin(hard_split), "key"].map(hard_split)
    data.loc[(data.split == "") & data.key.isin(forced_test), "split"] = "test"
    report["forced_to_test_previous_heldout_overlap"] = int(((data.split == "test") & data.key.isin(forced_test)).sum())

    rest = data[data.split == ""]
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, idx) in enumerate(folds.split(rest, rest.label, rest.group)):
        name = "test" if fold in TEST_FOLDS else "validation" if fold in VAL_FOLDS else "train"
        data.loc[rest.index[idx], "split"] = name

    splits = {name: data[data.split == name].reset_index(drop=True) for name in ("train", "validation", "test")}
    train_groups = set(splits["train"].group)
    report["splits"] = {
        name: {
            "rows": int(len(frame)),
            "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()},
            "rows_whose_registered_domain_is_in_train": int(frame.group.isin(train_groups).sum()) if name != "train" else None,
        }
        for name, frame in splits.items()
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = len(set(splits[a].key) & set(splits[b].key))
        if overlap:
            raise RuntimeError(f"URL leakage between {a} and {b}: {overlap} keys")
    report["url_overlap_between_splits"] = 0
    return splits, report


def main() -> None:
    seed_everything()
    torch.set_num_threads(4)
    splits, report = prepare_splits()
    print(json.dumps(report, indent=2))

    vocab = build_enhanced_vocab()
    (MODELS_DIR / "charcnn_bilstm_vocab.json").write_text(json.dumps(vocab, indent=2) + "\n", encoding="utf-8")

    train = splits["train"]
    hard_train = train[train.key.isin({canonical_key(c["url"]) for c in json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text()) if c["hard_negative_split"] == "train"})]
    train_urls = train.clean_url.tolist() + hard_train.clean_url.tolist() * (HARD_TRAIN_REPEAT - 1)
    train_labels = np.r_[train.label.to_numpy(np.float32), np.tile(hard_train.label.to_numpy(np.float32), HARD_TRAIN_REPEAT - 1)]

    x_train = encode_urls_bilstm(train_urls, vocab)
    x_val = encode_urls_bilstm(splits["validation"].clean_url, vocab)
    y_val = splits["validation"].label.to_numpy(int)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(x_train, dtype=torch.long), torch.as_tensor(train_labels)),
        batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED),
    )

    model = CharCNNBiLSTMAttention(vocab_size=len(vocab))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    checkpoint = MODELS_DIR / "charcnn_bilstm_best.pt"
    best_auc, patience, history = 0.0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        model.train()
        total = 0.0
        for seqs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(seqs), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item() * len(labels)
        val = compute_metrics(y_val, predict(model, x_val))
        scheduler.step(val["roc_auc"])
        history.append({"epoch": epoch, "train_loss": total / len(loader.dataset), "validation": val, "seconds": time.time() - start})
        print(f"epoch {epoch} loss={total / len(loader.dataset):.4f} val_auc={val['roc_auc']:.4f} val_f1={val['f1']:.4f} val_fpr={val['fpr']:.4f} ({time.time() - start:.0f}s)", flush=True)
        if val["roc_auc"] > best_auc:
            best_auc, patience = val["roc_auc"], 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    onnx_path = MODELS_DIR / "charcnn_bilstm.onnx"
    torch.onnx.export(
        model, torch.zeros((1, MAX_SEQUENCE_LENGTH), dtype=torch.long), str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )

    test = splits["test"]
    x_test = encode_urls_bilstm(test.clean_url, vocab)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = np.concatenate([session.run(["logits"], {"input": x_test[i:i + 512]})[0] for i in range(0, len(x_test), 512)])
    with torch.no_grad():
        torch_logits = np.concatenate([model(torch.as_tensor(x_test[i:i + 512])).numpy() for i in range(0, len(x_test), 512)])
    np.testing.assert_allclose(onnx_logits, torch_logits, rtol=1e-3, atol=1e-3)
    test_prob = 1.0 / (1.0 + np.exp(-onnx_logits))

    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text(encoding="utf-8"))
    hard_prob = 1.0 / (1.0 + np.exp(-session.run(["logits"], {"input": encode_urls_bilstm([c["url"] for c in hard], vocab)})[0]))
    hard_eval = {
        split: {"rows": int(sum(c["hard_negative_split"] == split for c in hard)),
                "false_positives": int(sum(p >= THRESHOLD for c, p in zip(hard, hard_prob) if c["hard_negative_split"] == split))}
        for split in ("train", "validation", "test")
    }

    metrics = {
        "model_version": MODEL_VERSION,
        "architecture": "CharCNN + BiLSTM + Attention",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "threshold": THRESHOLD,
        "data": report,
        "history": history,
        "selected_epoch": int(max(history, key=lambda h: h["validation"]["roc_auc"])["epoch"]),
        "validation": compute_metrics(y_val, predict(model, x_val)),
        "test": compute_metrics(test.label.to_numpy(int), test_prob),
        "hard_negatives": hard_eval,
        "onnx_parity_max_abs_diff": float(np.max(np.abs(onnx_logits - torch_logits))),
    }
    (RESULTS_DIR / "charcnn_bilstm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    test[["url", "clean_url", "label", "group"]].assign(probability=test_prob).to_csv(RESULTS_DIR / "charcnn_bilstm_test_set.csv", index=False)
    config = {
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "dataset_sha256": report["source_sha256"],
        "training_date": date.today().isoformat(),
        "preprocessing": "src.charcnn_bilstm.encode_url_bilstm (canonicalize_url, first 256 characters)",
    }
    (MODELS_DIR / "charcnn_bilstm_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("selected_epoch", "validation", "test", "hard_negatives")}, indent=2))


if __name__ == "__main__":
    main()
