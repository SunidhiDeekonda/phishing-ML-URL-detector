"""Tests for the CharCNN + BiLSTM + Attention production path."""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient

import app.inference as inference
from app.inference import URLInference
from app.main import app
from src.charcnn_bilstm import MAX_SEQUENCE_LENGTH, build_enhanced_vocab, encode_url_bilstm
from src.dataset_cleaning import clean_training_url
from src.url_normalization import canonicalize_url

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / inference.BILSTM_CONFIG).read_text())
CHECKPOINT = inference.BILSTM_ONNX.replace(".onnx", "_best.pt")


@pytest.fixture(scope="module")
def service() -> URLInference:
    value = URLInference()
    value.load_models()
    return value


def test_inference_path_runs_without_torch() -> None:
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "from app.inference import URLInference\n"
        "result = URLInference().predict('https://github.com/login')\n"
        "assert result['bilstm_probability'] is not None\n"
        "print(result['model_version'])\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert CONFIG["model_version"] in completed.stdout


@pytest.mark.parametrize("raw", [
    "votesmart.org/candidate/biography/39391",
    "WWW.Example.com/Path?q=1",
    "hyhpzx.cn/js/?http://",
    "https://github.com/login/",
    "  HTTPS://GITHUB.COM/  ",
])
def test_training_and_serving_encodings_match(raw: str) -> None:
    vocab = build_enhanced_vocab()
    assert np.array_equal(encode_url_bilstm(raw, vocab), encode_url_bilstm(canonicalize_url(raw), vocab))


def test_encoder_shape_padding_and_truncation() -> None:
    vocab = build_enhanced_vocab()
    short = encode_url_bilstm("a.com", vocab)
    long = encode_url_bilstm("https://a.com/" + "x" * 500, vocab)
    assert short.shape == long.shape == (MAX_SEQUENCE_LENGTH,)
    assert short.dtype == np.int64 and short[-1] == 0
    assert long[-1] != 0


def test_served_probability_is_unmodified_bilstm_output(service: URLInference) -> None:
    session = ort.InferenceSession(str(ROOT / inference.BILSTM_ONNX), providers=["CPUExecutionProvider"])
    vocab = json.loads((ROOT / inference.BILSTM_VOCAB).read_text())
    for url in ["https://github.com/login", "https://sites.google.com/view/account-verify", "http://paypal-verify.example.tk/login"]:
        result = service.predict(url)
        logit = session.run(["logits"], {"input": encode_url_bilstm(result["url"], vocab)[np.newaxis, :]})[0][0]
        assert result["phishing_probability"] == result["bilstm_probability"]
        assert result["phishing_probability"] == pytest.approx(1.0 / (1.0 + np.exp(-logit)), abs=1e-6)
        assert result["model_version"] == CONFIG["model_version"]
        assert result["verdict"] == ("PHISHING" if result["phishing_probability"] >= 0.5 else "LEGITIMATE")


def test_inference_has_no_domain_allowlist() -> None:
    source = inspect.getsource(inference).lower()
    assert "allowlist" not in source and "authentic_domain" not in source and "google.com" not in source


def test_mismatched_artifact_hash_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = dict(CONFIG, onnx_sha256="0" * 64)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(bad))
    monkeypatch.setattr(inference, "BILSTM_CONFIG", str(path))
    with pytest.raises(RuntimeError, match="sha256"):
        URLInference().load_models()


def test_onnx_matches_pytorch_checkpoint() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is only installed for training")
    # torch and LightGBM bundle conflicting OpenMP runtimes on macOS, so torch runs in its own process.
    code = (
        "import numpy as np, onnxruntime as ort, torch\n"
        "from src.charcnn_bilstm import build_enhanced_vocab, encode_url_bilstm\n"
        "from src.charcnn_bilstm_model import CharCNNBiLSTMAttention\n"
        "model = CharCNNBiLSTMAttention()\n"
        f"model.load_state_dict(torch.load('{CHECKPOINT}', map_location='cpu', weights_only=True))\n"
        "model.eval()\n"
        "vocab = build_enhanced_vocab()\n"
        "batch = np.stack([encode_url_bilstm(u, vocab) for u in ['https://github.com/login', 'http://1.2.3.4/a.php', 'example.org']])\n"
        f"session = ort.InferenceSession('{inference.BILSTM_ONNX}', providers=['CPUExecutionProvider'])\n"
        "with torch.no_grad():\n"
        "    expected = model(torch.as_tensor(batch)).numpy()\n"
        "np.testing.assert_allclose(session.run(['logits'], {'input': batch})[0], expected, atol=1e-4)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_health_and_predict_report_bilstm() -> None:
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["bilstm_loaded"] is True and health["model_version"] == CONFIG["model_version"]
    payload = client.post("/predict", json={"url": "https://github.com/login"}).json()
    assert 0.0 <= payload["bilstm_probability"] <= 1.0


@pytest.mark.parametrize("raw,expected", [
    ("'caledit.com/restore\\%3fcmd=_login'", ("caledit.com/restore%3fcmd=_login", "repaired")),
    ("a.b.co/x&amp;y=1", ("a.b.co/x&y=1", "repaired")),
    ("adelinabobrova1979@yandex.com", (None, "rejected_email_address")),
    ("fdldadakknbfaacf.org /", (None, "rejected_internal_whitespace")),
    ("cgi-bin/item1", (None, "rejected_invalid_host")),
    ("http://user:pw@host.com:8080/", ("http://user:pw@host.com:8080/", "kept")),
    ("ccr_2.tripod.com/bamboo/", ("ccr_2.tripod.com/bamboo/", "kept")),
])
def test_training_url_cleaning(raw: str, expected: tuple) -> None:
    assert clean_training_url(raw) == expected
