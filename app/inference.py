"""Inference helpers for the local FastAPI demo application.

This module loads models once and exposes deterministic prediction for URL inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from src.char_tokenizer import MAX_SEQUENCE_LENGTH, build_vocab, encode_url
from src.features import extract_url_features
from src.url_normalization import canonicalize_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_CNN_WEIGHT = 0.60
REFERENCE_LIGHTGBM_WEIGHT = 0.40
SELECTED_CNN_WEIGHT = 0.95
SELECTED_LIGHTGBM_WEIGHT = 0.05
THRESHOLD = 0.50
MAX_URL_LENGTH = 2048
MODEL_VERSION = "url-ensemble-production-v2.0"

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
class LightGBMOnnxModel:
    session: object
    calibration_method: str
    calibration_a: float
    calibration_b: float


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

        config_path = PROJECT_ROOT / "models/production_v2_config.json"
        production_v2_available = config_path.exists()
        if production_v2_available:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.cnn_weight = float(config["cnn_weight"])
            self.lightgbm_weight = float(config["lightgbm_weight"])
        else:
            self.cnn_weight = SELECTED_CNN_WEIGHT
            self.lightgbm_weight = SELECTED_LIGHTGBM_WEIGHT

        lightgbm_path = PROJECT_ROOT / ("models/lightgbm_production_v2.onnx" if production_v2_available else "models/lightgbm.onnx")
        if not lightgbm_path.exists():
            raise FileNotFoundError(f"Missing model file: {lightgbm_path}")
        calibration_path = PROJECT_ROOT / ("models/lightgbm_production_v2_calibration.json" if production_v2_available else "models/lightgbm_calibration.json")
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {calibration_path}")
        try:
            import onnxruntime as ort

            lightgbm_session = ort.InferenceSession(
                str(lightgbm_path), providers=["CPUExecutionProvider"]
            )
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration.get("method") not in {"sigmoid", "logistic_logit"}:
                raise ValueError("Unsupported LightGBM calibration method")
            lightgbm_model = LightGBMOnnxModel(
                session=lightgbm_session,
                calibration_method=str(calibration["method"]),
                calibration_a=float(calibration["a"]),
                calibration_b=float(calibration["b"]),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load portable LightGBM artifacts: {type(exc).__name__}: {exc}"
            ) from exc

        cnn_path = PROJECT_ROOT / ("models/char_cnn_production_v2.onnx" if production_v2_available else "models/char_cnn.onnx")
        if not cnn_path.exists():
            raise FileNotFoundError(f"Missing model file: {cnn_path}")

        try:
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
        model = bundle.lightgbm_model
        model_input = features_df[bundle.feature_columns].to_numpy(dtype=np.float32)
        raw_scores = model.session.run(
            ["probabilities"], {"input": model_input}
        )[0]
        raw_score = float(raw_scores[0][1])
        calibration_input = raw_score
        if model.calibration_method == "logistic_logit":
            clipped = min(max(raw_score, 1e-6), 1 - 1e-6)
            calibration_input = float(np.log(clipped / (1 - clipped)))
        calibrated_logit = model.calibration_a * calibration_input + model.calibration_b
        if model.calibration_method == "sigmoid":
            if calibrated_logit >= 0:
                exp_negative = float(np.exp(-calibrated_logit))
                return exp_negative / (1.0 + exp_negative)
            return 1.0 / (1.0 + float(np.exp(calibrated_logit)))
        if calibrated_logit >= 0:
            exp_negative = float(np.exp(-calibrated_logit))
            return 1.0 / (1.0 + exp_negative)
        exp_positive = float(np.exp(calibrated_logit))
        return exp_positive / (1.0 + exp_positive)

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
        normalized_url = canonicalize_url(url)
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
        selected_ensemble_probability = self.cnn_weight * cnn_probability + self.lightgbm_weight * lightgbm_probability

        phishing_probability = selected_ensemble_probability
        verdict = "PHISHING" if phishing_probability >= THRESHOLD else "LEGITIMATE"
        confidence = phishing_probability if verdict == "PHISHING" else (1 - phishing_probability)

        important_features = _extract_signals(url_features)
        notable_signals = []
        if verdict == "PHISHING":
            checks = [
                (url_features["suspicious_tld"] > 0, "The top-level domain is in the model's suspicious-TLD feature list."),
                (url_features["is_ip_host"] > 0, "The hostname is an IP address."),
                (url_features["has_at_symbol"] > 0, "The URL contains an @ symbol."),
                (url_features["url_length"] >= 60, f"The URL is relatively long ({int(url_features['url_length'])} characters)."),
                (url_features["path_length"] >= 25, f"The path is relatively long ({int(url_features['path_length'])} characters)."),
                (url_features["hyphen_count"] >= 3, f"The URL contains several hyphens ({int(url_features['hyphen_count'])})."),
                (url_features["token_login"] > 0, "The URL text contains the token 'login'."),
                (url_features["token_verify"] > 0, "The URL text contains the token 'verify'."),
                (url_features["token_account"] > 0, "The URL text contains the token 'account'."),
            ]
            notable_signals = [message for matched, message in checks if matched][:5]

        return {
            "url": normalized_url,
            "model_version": MODEL_VERSION,
            "verdict": verdict,
            "phishing_probability": phishing_probability,
            "confidence": confidence,
            "cnn_probability": cnn_probability,
            "lightgbm_probability": lightgbm_probability,
            "reference_ensemble_probability": reference_ensemble_probability,
            "selected_ensemble_probability": selected_ensemble_probability,
            "selected_weights": {
                "cnn": self.cnn_weight,
                "lightgbm": self.lightgbm_weight,
            },
            "reference_weights": {
                "cnn": REFERENCE_CNN_WEIGHT,
                "lightgbm": REFERENCE_LIGHTGBM_WEIGHT,
            },
            "threshold": THRESHOLD,
            "important_features": important_features,
            "notable_signals": notable_signals,
            "explanation_caveat": "Notable values are descriptive signals, not proof of causation. This probabilistic prediction can produce false positives.",
        }


_URL_INFERENCE = URLInference()


def get_inference_service() -> URLInference:
    return _URL_INFERENCE
