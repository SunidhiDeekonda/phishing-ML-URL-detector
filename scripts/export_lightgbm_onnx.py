"""Export the frozen calibrated LightGBM model to portable ONNX artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import onnxmltools
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MODEL_PATH = PROJECT_ROOT / "models/lightgbm_raw.pkl"
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "models/lightgbm_calibrated.pkl"
ONNX_MODEL_PATH = PROJECT_ROOT / "models/lightgbm.onnx"
CALIBRATION_PATH = PROJECT_ROOT / "models/lightgbm_calibration.json"
FEATURES_PATH = PROJECT_ROOT / "data/processed/features.csv"
MAX_ABSOLUTE_ERROR = 5e-4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibrate(raw_scores: np.ndarray, a: float, b: float) -> np.ndarray:
    logits = a * raw_scores + b
    result = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    exp_negative = np.exp(-logits[positive])
    result[positive] = exp_negative / (1.0 + exp_negative)
    result[~positive] = 1.0 / (1.0 + np.exp(logits[~positive]))
    return result


def main() -> None:
    raw_model = joblib.load(RAW_MODEL_PATH)
    calibrated_model = joblib.load(CALIBRATED_MODEL_PATH)
    calibrated_classifiers = calibrated_model.calibrated_classifiers_
    if len(calibrated_classifiers) != 1:
        raise RuntimeError("Expected exactly one calibrated LightGBM classifier")
    calibrated_classifier = calibrated_classifiers[0]
    if calibrated_classifier.method != "sigmoid" or len(calibrated_classifier.calibrators) != 1:
        raise RuntimeError("Expected one sigmoid calibrator")
    calibrator = calibrated_classifier.calibrators[0]
    calibration_a = float(calibrator.a_)
    calibration_b = float(calibrator.b_)

    feature_count = int(raw_model.booster_.num_feature())
    onnx_model = onnxmltools.convert_lightgbm(
        raw_model,
        initial_types=[("input", FloatTensorType([None, feature_count]))],
        target_opset=15,
    )
    tree_nodes = [node for node in onnx_model.graph.node if node.op_type == "TreeEnsembleClassifier"]
    if len(tree_nodes) != 1:
        raise RuntimeError("Expected one ONNX TreeEnsembleClassifier node")
    post_transform = [
        attribute for attribute in tree_nodes[0].attribute if attribute.name == "post_transform"
    ]
    if len(post_transform) != 1:
        raise RuntimeError("Missing ONNX post_transform attribute")
    post_transform[0].s = b"NONE"
    ONNX_MODEL_PATH.write_bytes(onnx_model.SerializeToString())

    metadata = {
        "method": "sigmoid",
        "a": calibration_a,
        "b": calibration_b,
        "positive_class": 1,
        "feature_count": feature_count,
        "source_raw_model_sha256": _sha256(RAW_MODEL_PATH),
        "source_calibrated_model_sha256": _sha256(CALIBRATED_MODEL_PATH),
    }
    CALIBRATION_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    feature_frame = pd.read_csv(FEATURES_PATH)
    model_input = feature_frame.to_numpy(dtype=np.float32)
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    score_maps = session.run(["probabilities"], {"input": model_input})[0]
    raw_scores = np.asarray([scores[1] for scores in score_maps], dtype=np.float64)
    observed = _calibrate(raw_scores, calibration_a, calibration_b)
    expected = calibrated_model.predict_proba(feature_frame)[:, 1]
    max_error = float(np.max(np.abs(expected - observed)))
    disagreements = int(np.sum((expected >= 0.5) != (observed >= 0.5)))
    if max_error > MAX_ABSOLUTE_ERROR or disagreements:
        raise RuntimeError(
            f"ONNX verification failed: max_error={max_error}, disagreements={disagreements}"
        )
    print(f"Exported {ONNX_MODEL_PATH}")
    print(f"Maximum calibrated probability error: {max_error:.12g}")
    print(f"Classification disagreements: {disagreements}")


if __name__ == "__main__":
    main()
