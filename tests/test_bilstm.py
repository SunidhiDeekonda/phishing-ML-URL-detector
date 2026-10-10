"""Tests for the modern CharCNN + BiLSTM + Attention model and Laya integration.

Validates that real-world legitimate URLs with deep paths, login paths, and
security keywords are correctly classified as LEGITIMATE with high confidence,
fixing the 50% false positive flaw of the legacy model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
MODELS_DIR = ROOT / "models"

from src.charcnn_bilstm import (
    CharCNNBiLSTMAttention,
    MAX_SEQUENCE_LENGTH,
    build_enhanced_vocab,
    encode_url_bilstm,
)
from src.laya_detector import is_laya_available, analyze_url_with_laya


@pytest.fixture(scope="module")
def onnx_session():
    onnx_path = MODELS_DIR / "charcnn_bilstm.onnx"
    if not onnx_path.exists():
        pytest.skip("charcnn_bilstm.onnx not yet trained/exported.")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def vocab():
    vocab_path = MODELS_DIR / "charcnn_bilstm_vocab.json"
    if vocab_path.exists():
        return json.loads(vocab_path.read_text(encoding="utf-8"))
    return build_enhanced_vocab()


from src.domain_reputation import calibrate_probability_with_domain

def predict_onnx(url: str, session, vocab) -> dict:
    encoded = encode_url_bilstm(url, vocab, max_len=MAX_SEQUENCE_LENGTH)
    input_data = encoded[np.newaxis, :]
    logits = session.run(["logits"], {"input": input_data})[0]
    logit = float(logits[0])
    raw_prob = float(1.0 / (1.0 + np.exp(-logit)))
    prob = calibrate_probability_with_domain(url, raw_prob)
    verdict = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    return {
        "url": url,
        "probability": prob,
        "verdict": verdict,
    }


def test_hard_negative_legitimate_urls(onnx_session, vocab):
    """The critical test: real-world legitimate URLs must NOT be flagged as phishing."""
    critical_legit_urls = [
        "https://github.com/login",
        "https://vercel.com/login",
        "https://en.wikipedia.org/wiki/Phishing",
        "https://support.apple.com/en-us/102660",
        "https://help.openai.com/en/articles/6614161-how-can-i-contact-support",
        "https://docs.python.org/3/library/functions.html",
        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication",
        "https://doc.rust-lang.org/book/ch01-01-installation.html",
        "https://dash.cloudflare.com/login",
        "https://myaccount.google.com/security",
        "https://platform.openai.com/login",
        "https://account.apple.com/sign-in",
        "https://owasp.org/www-community/attacks/Phishing",
        "https://github.com/SunidhiDeekonda/phishing-ML-URL-detector",
    ]

    failures = []
    for url in critical_legit_urls:
        res = predict_onnx(url, onnx_session, vocab)
        if res["verdict"] != "LEGITIMATE":
            failures.append(f"{url} -> {res['verdict']} (prob={res['probability']:.4f})")

    assert not failures, f"False positives detected on legitimate sites: {failures}"


def test_obvious_phishing_urls(onnx_session, vocab):
    """Obvious phishing URLs must be flagged as PHISHING."""
    obvious_phish_urls = [
        "http://suspicious-login-verify-account-paypal.com/secure/login.php?id=123",
        "http://107.160.185.171/office/austin/server/cp.php?m=login",
        "http://ctts.in/assets/download/revalidate-alert/draft/up.php",
        "http://paypal-verification-alert-account-recovery.tk/login.html",
    ]

    for url in obvious_phish_urls:
        res = predict_onnx(url, onnx_session, vocab)
        assert res["verdict"] == "PHISHING", f"Phishing URL missed: {url} (prob={res['probability']:.4f})"
        assert res["probability"] > 0.70


def test_laya_adapter_fallback():
    """Verify that Laya adapter functions cleanly even when model weights are absent or loaded."""
    res = analyze_url_with_laya("https://example.com")
    assert isinstance(res, dict)
    assert "available" in res


def test_api_predict_charcnn_bilstm_integration():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    # Critical test: github.com/login must be legitimate
    resp_login = client.post("/predict", json={"url": "https://github.com/login"})
    assert resp_login.status_code == 200
    data = resp_login.json()
    assert data["verdict"] == "LEGITIMATE"
    assert data["bilstm_probability"] is not None
    assert data["bilstm_probability"] < 0.20

    # Obvious phishing URL must be phishing
    resp_phish = client.post("/predict", json={"url": "http://suspicious-login-verify-account-paypal.com/secure/login.php?id=123"})
    assert resp_phish.status_code == 200
    data_phish = resp_phish.json()
    assert data_phish.verdict == "PHISHING" if hasattr(data_phish, 'verdict') else data_phish["verdict"] == "PHISHING"
    assert data_phish["phishing_probability"] > 0.80


def test_api_predict_deep_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.post("/predict-deep", json={"url": "https://github.com/login"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "LEGITIMATE"
    assert "fast_prediction" in data
    assert "laya_analysis" in data
    assert data["method"] in {"hybrid_laya_bilstm", "charcnn_bilstm_edge"}

