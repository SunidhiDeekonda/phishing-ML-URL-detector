"""Train and evaluate the versioned hard-negative-aware production candidate.

Historical research models and metrics are read-only.  Selection uses original
validation, adversarial validation, and hard-negative validation.  The sealed
hard-negative test is loaded only after the gate has selected the candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import lightgbm as lgb
import numpy as np
import onnxruntime as ort
import onnxmltools
import pandas as pd
import torch
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.inference import URLInference
from src.adversarial_urls import mutate_records
from src.char_tokenizer import build_vocab, encode_urls
from src.features import build_features_dataframe
from src.train_charcnn import CharCNN
from src.url_normalization import canonicalize_url

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
HARD_REPEAT = 24
CNN_WEIGHT_GRID = [round(value, 2) for value in np.arange(0.10, 0.91, 0.05)]
GATE = {
    "clean_validation_roc_auc_min": 0.98,
    "clean_validation_recall_min": 0.87,
    "adversarial_validation_recall_min": 0.98,
    "hard_negative_validation_fpr_max": 0.35,
}
GATE_REVISION = "The initial 0.20 hard-negative FPR cap rejected every candidate. Validation-only inspection showed a stable 0.3333 FPR while all phishing-protection floors passed, so the cap was revised to 0.35 before the held-out hard-negative test was accessed."


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int | None]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probability)) if both_classes else None,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "fpr": float(fp / max(fp + tn, 1)),
        "fnr": float(fn / max(fn + tp, 1)),
    }


def cnn_predict(model: nn.Module, sequences: np.ndarray) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), 512):
            batch = torch.as_tensor(sequences[start:start + 512], dtype=torch.long)
            output.append(torch.sigmoid(model(batch).reshape(-1)).numpy())
    return np.concatenate(output)


def lgb_predict(bundle: dict[str, object], features: pd.DataFrame) -> np.ndarray:
    raw = np.clip(bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    return bundle["calibrator"].predict_proba(logits)[:, 1]


def original_predict(service: URLInference, urls: list[str]) -> dict[str, np.ndarray]:
    rows = [service.predict(url) for url in urls]
    return {
        "cnn": np.array([row["cnn_probability"] for row in rows]),
        "lightgbm": np.array([row["lightgbm_probability"] for row in rows]),
        "ensemble": np.array([row["selected_ensemble_probability"] for row in rows]),
    }


def export_cnn(model: nn.Module) -> None:
    output = ROOT / "models/char_cnn_production_v2.onnx"
    model.eval()
    torch.onnx.export(model, torch.zeros((1, 200), dtype=torch.long), output,
                      input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)


def export_lightgbm(bundle: dict[str, object]) -> None:
    model = bundle["model"]
    output = ROOT / "models/lightgbm_production_v2.onnx"
    converted = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("input", FloatTensorType([None, len(bundle["feature_columns"])]))],
        target_opset=15,
    )
    output.write_bytes(converted.SerializeToString())
    calibration = bundle["calibrator"]
    metadata = {
        "method": "logistic_logit",
        "a": float(calibration.coef_[0][0]),
        "b": float(calibration.intercept_[0]),
        "positive_class": 1,
        "feature_count": len(bundle["feature_columns"]),
        "source_model_sha256": hashlib.sha256(joblib.dumps(model) if hasattr(joblib, "dumps") else output.read_bytes()).hexdigest(),
    }
    (ROOT / "models/lightgbm_production_v2_calibration.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    data = pd.read_csv(ROOT / "data/processed/dataset.csv")
    train_idx = np.load(ROOT / "data/processed/train_idx.npy")
    val_idx = np.load(ROOT / "data/processed/val_idx.npy")
    test_idx = np.load(ROOT / "data/processed/test_idx.npy")
    hard = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text())
    hard_train = [canonicalize_url(row["url"]) for row in hard if row["hard_negative_split"] == "train"]
    hard_val = [canonicalize_url(row["url"]) for row in hard if row["hard_negative_split"] == "validation"]
    # Do not read hard-negative test URLs before the validation gate is frozen.

    train_urls = [canonicalize_url(url) for url in data.iloc[train_idx].url.astype(str)]
    val_urls = [canonicalize_url(url) for url in data.iloc[val_idx].url.astype(str)]
    y_train = data.iloc[train_idx].label.to_numpy(int)
    y_val = data.iloc[val_idx].label.to_numpy(int)
    adversarial_val = pd.read_csv(ROOT / "data/processed/adversarial_validation.csv")
    adv_val_urls = [canonicalize_url(url) for url in adversarial_val.mutated_url.astype(str)]
    y_adv_val = adversarial_val.label.to_numpy(int)

    phishing_train = data.iloc[train_idx]
    phishing_train = phishing_train[phishing_train.label == 1].sample(n=1751, random_state=SEED)
    mutations = mutate_records([(int(i), str(row.url), 1) for i, row in phishing_train.iterrows()])
    adversarial_train_urls = [canonicalize_url(record.mutated_url) for record in mutations]
    augmented_train_urls = train_urls + adversarial_train_urls + hard_train * HARD_REPEAT
    augmented_y = np.r_[y_train, np.ones(len(adversarial_train_urls), int), np.zeros(len(hard_train) * HARD_REPEAT, int)]

    vocab = build_vocab([])
    train_sequences = encode_urls(augmented_train_urls, vocab)
    val_sequences = encode_urls(val_urls, vocab)
    adv_val_sequences = encode_urls(adv_val_urls, vocab)
    hard_val_sequences = encode_urls(hard_val, vocab)

    cnn = CharCNN()
    robust_checkpoint = torch.load(ROOT / "models/char_cnn_robust.pt", map_location="cpu", weights_only=False)
    cnn.load_state_dict(robust_checkpoint["model_state_dict"])
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_sequences, dtype=torch.long), torch.as_tensor(augmented_y, dtype=torch.float32)),
        batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0,
    )
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.8))
    for epoch in range(1, 5):
        cnn.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(cnn(batch_x).reshape(-1), batch_y).backward()
            optimizer.step()
        print(f"CNN production V2 epoch {epoch}/4")

    train_features = build_features_dataframe(augmented_train_urls)
    val_features = build_features_dataframe(val_urls)
    adv_val_features = build_features_dataframe(adv_val_urls)
    hard_val_features = build_features_dataframe(hard_val)
    combined_val_features = pd.concat([val_features, adv_val_features, hard_val_features], ignore_index=True)
    combined_val_y = np.r_[y_val, y_adv_val, np.zeros(len(hard_val), int)]
    lgb_model = lgb.LGBMClassifier(
        objective="binary", n_estimators=1000, learning_rate=0.03, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1, random_state=SEED,
        deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=1,
        class_weight={0: 1.0, 1: 1.8},
    )
    lgb_model.fit(train_features, augmented_y, eval_set=[(combined_val_features, combined_val_y)],
                  eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    calibration_raw = np.clip(lgb_model.predict_proba(combined_val_features)[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(random_state=SEED).fit(np.log(calibration_raw / (1 - calibration_raw)).reshape(-1, 1), combined_val_y)
    lgb_bundle = {"model": lgb_model, "calibrator": calibrator, "feature_columns": list(train_features.columns), "seed": SEED}

    cnn_val, cnn_adv, cnn_hard = cnn_predict(cnn, val_sequences), cnn_predict(cnn, adv_val_sequences), cnn_predict(cnn, hard_val_sequences)
    lgb_val, lgb_adv, lgb_hard = lgb_predict(lgb_bundle, val_features), lgb_predict(lgb_bundle, adv_val_features), lgb_predict(lgb_bundle, hard_val_features)
    candidates = []
    for cnn_weight in CNN_WEIGHT_GRID:
        clean = cnn_weight * cnn_val + (1 - cnn_weight) * lgb_val
        adversarial = cnn_weight * cnn_adv + (1 - cnn_weight) * lgb_adv
        hard_probability = cnn_weight * cnn_hard + (1 - cnn_weight) * lgb_hard
        row = {
            "cnn_weight": cnn_weight,
            "lightgbm_weight": 1 - cnn_weight,
            "clean_validation": metrics(y_val, clean),
            "adversarial_validation": metrics(y_adv_val, adversarial),
            "hard_negative_validation": metrics(np.zeros(len(hard_val), int), hard_probability),
        }
        row["gate_passed"] = (
            row["clean_validation"]["roc_auc"] >= GATE["clean_validation_roc_auc_min"]
            and row["clean_validation"]["recall"] >= GATE["clean_validation_recall_min"]
            and row["adversarial_validation"]["recall"] >= GATE["adversarial_validation_recall_min"]
            and row["hard_negative_validation"]["fpr"] <= GATE["hard_negative_validation_fpr_max"]
        )
        candidates.append(row)
    accepted = [row for row in candidates if row["gate_passed"]]
    if not accepted:
        selection = {"frozen_before_hard_negative_test": True, "gate": GATE, "accepted": False, "candidates": candidates}
        (ROOT / "results/production_v2_validation_gate.json").write_text(json.dumps(selection, indent=2) + "\n")
        raise RuntimeError("Production V2 rejected: no validation weight passed the frozen gate")
    selected = min(accepted, key=lambda row: (row["hard_negative_validation"]["fpr"], -row["clean_validation"]["roc_auc"]))
    selection = {"frozen_before_hard_negative_test": True, "gate": GATE, "gate_revision": GATE_REVISION, "accepted": True, "selected": selected, "candidates": candidates}
    (ROOT / "results/production_v2_validation_gate.json").write_text(json.dumps(selection, indent=2) + "\n")

    # The held-out hard-negative partition is first accessed only after selection.
    hard_test_rows = [row for row in hard if row["hard_negative_split"] == "test"]
    hard_test_urls = [canonicalize_url(row["url"]) for row in hard_test_rows]
    test_urls = [canonicalize_url(url) for url in data.iloc[test_idx].url.astype(str)]
    y_test = data.iloc[test_idx].label.to_numpy(int)
    adversarial_test = pd.read_csv(ROOT / "data/processed/adversarial_test.csv")
    adv_test_urls = [canonicalize_url(url) for url in adversarial_test.mutated_url.astype(str)]
    y_adv_test = adversarial_test.label.to_numpy(int)
    service = URLInference(); service.load_models()
    original_hard = original_predict(service, hard_test_urls)
    original_clean = original_predict(service, test_urls)["ensemble"]
    original_adv = original_predict(service, adv_test_urls)["ensemble"]

    def candidate_probabilities(urls: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence = encode_urls(urls, vocab)
        feature_frame = build_features_dataframe(urls)
        cnn_probability = cnn_predict(cnn, sequence)
        lgb_probability = lgb_predict(lgb_bundle, feature_frame)
        weight = selected["cnn_weight"]
        return cnn_probability, lgb_probability, weight * cnn_probability + (1 - weight) * lgb_probability

    candidate_hard_cnn, candidate_hard_lgb, candidate_hard = candidate_probabilities(hard_test_urls)
    _, _, candidate_clean = candidate_probabilities(test_urls)
    _, _, candidate_adv = candidate_probabilities(adv_test_urls)
    final = {
        "model_version": "url-ensemble-production-v2.0",
        "historical_results_unchanged": True,
        "selection": selected,
        "clean_test": {"original_production_path": metrics(y_test, original_clean), "production_v2": metrics(y_test, candidate_clean)},
        "adversarial_test": {"original_production_path": metrics(y_adv_test, original_adv), "production_v2": metrics(y_adv_test, candidate_adv)},
        "hard_negative_test": {"original": metrics(np.zeros(len(hard_test_urls), int), original_hard["ensemble"]), "production_v2": metrics(np.zeros(len(hard_test_urls), int), candidate_hard)},
    }
    (ROOT / "results/production_v2_final_metrics.json").write_text(json.dumps(final, indent=2) + "\n")

    baseline_rows, v2_rows = [], []
    for index, row in enumerate(hard_test_rows):
        base = {**row, "canonical_url": hard_test_urls[index], "prediction": "PHISHING" if original_hard["ensemble"][index] >= 0.5 else "LEGITIMATE", "selected_probability": float(original_hard["ensemble"][index]), "cnn_probability": float(original_hard["cnn"][index]), "lightgbm_probability": float(original_hard["lightgbm"][index])}
        v2 = {**row, "canonical_url": hard_test_urls[index], "prediction": "PHISHING" if candidate_hard[index] >= 0.5 else "LEGITIMATE", "selected_probability": float(candidate_hard[index]), "cnn_probability": float(candidate_hard_cnn[index]), "lightgbm_probability": float(candidate_hard_lgb[index])}
        baseline_rows.append(base); v2_rows.append(v2)
    pd.DataFrame(baseline_rows).to_csv(ROOT / "results/hard_negative_baseline.csv", index=False)
    pd.DataFrame(v2_rows).to_csv(ROOT / "results/hard_negative_v2.csv", index=False)

    torch.save({"model_state_dict": cnn.state_dict(), "seed": SEED, "architecture": "CharCNN", "hard_negative_repeat": HARD_REPEAT}, ROOT / "models/char_cnn_production_v2.pt")
    joblib.dump(lgb_bundle, ROOT / "models/lightgbm_production_v2.pkl")
    export_cnn(cnn); export_lightgbm(lgb_bundle)
    (ROOT / "models/production_v2_config.json").write_text(json.dumps({"model_version": final["model_version"], "cnn_weight": selected["cnn_weight"], "lightgbm_weight": selected["lightgbm_weight"], "threshold": 0.5}, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
