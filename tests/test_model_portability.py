from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from app.inference import URLInference


def test_lightgbm_onnx_matches_calibrated_pickle() -> None:
    feature_frame = pd.read_csv("data/processed/features.csv").iloc[:256]
    expected_model = joblib.load("models/lightgbm_calibrated.pkl")
    expected = expected_model.predict_proba(feature_frame)[:, 1]

    service = URLInference()
    bundle = service.load_models()
    observed = np.asarray(
        [service._predict_lgbm(feature_frame.iloc[[index]], bundle) for index in range(len(feature_frame))]
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=5e-4)
    np.testing.assert_array_equal(observed >= 0.5, expected >= 0.5)
