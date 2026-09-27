from __future__ import annotations

import json

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from app.inference import URLInference


def test_historical_lightgbm_onnx_matches_historical_pickle() -> None:
    feature_frame = pd.read_csv("data/processed/features.csv").iloc[:256]
    expected_model = joblib.load("models/lightgbm_calibrated.pkl")
    expected = expected_model.predict_proba(feature_frame)[:, 1]
    session = ort.InferenceSession("models/lightgbm.onnx", providers=["CPUExecutionProvider"])
    calibration = json.loads(open("models/lightgbm_calibration.json").read())
    score_maps = session.run(["probabilities"], {"input": feature_frame.to_numpy(dtype=np.float32)})[0]
    raw = np.asarray([scores[1] for scores in score_maps], dtype=float)
    logit = calibration["a"] * raw + calibration["b"]
    observed = 1.0 / (1.0 + np.exp(logit))

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5e-4)
    np.testing.assert_array_equal(observed >= 0.5, expected >= 0.5)


def test_production_v2_lightgbm_onnx_matches_v2_pickle() -> None:
    feature_frame = pd.read_csv("data/processed/features.csv").iloc[:256]
    expected_bundle = joblib.load("models/lightgbm_production_v2.pkl")
    raw = np.clip(expected_bundle["model"].predict_proba(feature_frame)[:, 1], 1e-6, 1 - 1e-6)
    expected = expected_bundle["calibrator"].predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]

    service = URLInference()
    bundle = service.load_models()
    observed = np.asarray(
        [service._predict_lgbm(feature_frame.iloc[[index]], bundle) for index in range(len(feature_frame))]
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5e-4)
    np.testing.assert_array_equal(observed >= 0.5, expected >= 0.5)
