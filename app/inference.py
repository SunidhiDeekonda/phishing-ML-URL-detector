"""Inference helpers for the local FastAPI demo application.

This module loads models once and exposes deterministic prediction for URL inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np
import pandas as pd

from src.char_tokenizer import MAX_SEQUENCE_LENGTH, build_vocab, encode_url, normalize_url
from src.features import extract_url_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CNN_WEIGHT = 0.60
REFERENCE_LIGHTGBM_WEIGHT = 0.40
SELECTED_CNN_WEIGHT = 0.95
SELECTED_LIGHTGBM_WEIGHT = 0.05
THRESHOLD = 0.50
MAX_URL_LENGTH = 2048

def _prepare_feature_columns() -> list[str]:
    features_path = PROJECT_ROOT / "data/processed/features.csv"
    feature_columns = list(pd.read_csv(features_path).columns)
    if len(feature_columns) != 36:
        raise RuntimeError(f"Expected 36 engineered features, found {len(feature_columns)}")
    return feature_columns


def _extract_signals(features: Mapping[str, float]) -> Dict[str, float | str]:
    return {
        "URL Length": int(features["url_length"]),
        "Host Length": int(features["host_length"]),
        "Hostname Entropy": float(features["hostname_entropy"]),
        "Number of Dots": int(features["num_dots"]),
        "Digit/Letter Ratio": float(features["digit_letter_ratio"]),
        "Number of Dots in Path": int(features["num_path_segments"]),
        "Suspicious TLD": bool(features["suspicious_tld"] > 0.5),
        "Contains Login Keyword": bool(features["token_login"] > 0.5),
        "Contains Verify Keyword": bool(features["token_verify"] > 0.5),
    }


@dataclass
class ModelBundle:
    lightgbm_model: object
    cnn_model: object
    cnn_model_loaded: bool
    lightgbm_loaded: bool
    vocab: Dict[str, int]
    feature_columns: list[str]
    device: str


class URLInference:
    def __init__(self) -> None:
        self._bundle: ModelBundle | None = None

    def load_models(self) -> ModelBundle:
        if self._bundle is not None:
            return self._bundle

        feature_columns = _prepare_feature_columns()
        vocab = build_vocab([])

        lightgbm_path = PROJECT_ROOT / "models/lightgbm_calibrated.pkl"
        if not lightgbm_path.exists():
            raise FileNotFoundError(f"Missing model file: {lightgbm_path}")
        try:
            lightgbm_model = joblib.load(lightgbm_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {lightgbm_path}") from exc

        cnn_path = PROJECT_ROOT / "models/char_cnn.onnx"
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing model file: {cnn_path}")

        try:
            import onnxruntime as ort

            cnn_model = ort.InferenceSession(
                str(cnn_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load {cnn_path}") from exc

        self._bundle = ModelBundle(
            lightgbm_model=lightgbm_model,
            cnn_model=cnn_model,
            cnn_model_loaded=True,
            lightgbm_loaded=True,
            vocab=vocab,
            feature_columns=feature_columns,
            device="cpu",
        )
        return self._bundle

    def _predict_lgbm(self, features_df: pd.DataFrame, bundle: ModelBundle) -> float:
        if bundle.lightgbm_model is None:
            raise RuntimeError("LightGBM model is not loaded")
        probs = bundle.lightgbm_model.predict_proba(features_df[bundle.feature_columns])[:, 1]
        return float(np.asarray(probs, dtype=np.float64)[0])

    def _predict_cnn(self, sequence: np.ndarray, bundle: ModelBundle) -> float:
        if bundle.cnn_model is None:
            raise RuntimeError("Char-CNN model is not loaded")
        if sequence.shape != (MAX_SEQUENCE_LENGTH,):
            raise ValueError(f"Expected sequence shape ({MAX_SEQUENCE_LENGTH},), got {sequence.shape}")

        logits = bundle.cnn_model.run(
            ["logits"], {"input_ids": sequence[np.newaxis, :].astype(np.int64)}
        )[0]
        logit = float(np.ravel(logits)[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict(self, url: str) -> Dict[str, object]:
        bundle = self.load_models()
        normalized_url = normalize_url(url)
        if not normalized_url:
            raise ValueError("URL must not be empty")
        if len(normalized_url) > MAX_URL_LENGTH:
            raise ValueError(f"URL must be <= {MAX_URL_LENGTH} characters")

        url_features = extract_url_features(normalized_url)
        features_df = pd.DataFrame([url_features], columns=bundle.feature_columns)

        if np.any(~np.isfinite(np.asarray(features_df[bundle.feature_columns], dtype=np.float64))):
            raise RuntimeError("Feature matrix contains non-finite values")

        lightgbm_probability = self._predict_lgbm(features_df, bundle)
        if not np.isfinite(lightgbm_probability):
            raise RuntimeError("Invalid LightGBM probability output")

        sequence = encode_url(normalized_url, bundle.vocab, max_len=MAX_SEQUENCE_LENGTH)
        cnn_probability = self._predict_cnn(sequence, bundle)
        if not np.isfinite(cnn_probability):
            raise RuntimeError("Invalid CNN probability output")

        reference_ensemble_probability = (
            REFERENCE_CNN_WEIGHT * cnn_probability
            + REFERENCE_LIGHTGBM_WEIGHT * lightgbm_probability
        )
        selected_ensemble_probability = (
            SELECTED_CNN_WEIGHT * cnn_probability
            + SELECTED_LIGHTGBM_WEIGHT * lightgbm_probability
        )

        phishing_probability = selected_ensemble_probability
        verdict = "PHISHING" if phishing_probability >= THRESHOLD else "LEGITIMATE"
        confidence = phishing_probability if verdict == "PHISHING" else (1 - phishing_probability)

        important_features = _extract_signals(url_features)

        return {
            "url": normalized_url,
            "verdict": verdict,
            "phishing_probability": phishing_probability,
            "confidence": confidence,
            "cnn_probability": cnn_probability,
            "lightgbm_probability": lightgbm_probability,
            "reference_ensemble_probability": reference_ensemble_probability,
            "selected_ensemble_probability": selected_ensemble_probability,
            "selected_weights": {
                "cnn": SELECTED_CNN_WEIGHT,
                "lightgbm": SELECTED_LIGHTGBM_WEIGHT,
            },
            "reference_weights": {
                "cnn": REFERENCE_CNN_WEIGHT,
                "lightgbm": REFERENCE_LIGHTGBM_WEIGHT,
            },
            "threshold": THRESHOLD,
            "important_features": important_features,
        }


_URL_INFERENCE = URLInference()


def get_inference_service() -> URLInference:
    return _URL_INFERENCE
