from __future__ import annotations

import inspect
import json
import socket

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.inference import URLInference
from app.main import app
from scripts.run_external_url_audit import load_cases, membership_context, registered_domain
from src.context_features import analyze_context
from src.url_normalization import canonicalize_url


@pytest.mark.parametrize("left,right,expected", [
    ("https://www.google.com", "https://www.google.com/", "https://www.google.com"),
    ("  HTTPS://WWW.GOOGLE.COM/  ", "www.google.com", "https://www.google.com"),
    ("https://github.com", "github.com/", "https://github.com"),
    ("https://openai.com/", " OPENAI.COM ", "https://openai.com"),
])
def test_equivalent_root_canonicalization(left, right, expected):
    assert canonicalize_url(left) == canonicalize_url(right) == expected


def test_canonicalization_preserves_meaningful_components():
    assert canonicalize_url(" HTTPS://EXAMPLE.COM/App/?Q=One#Part ") == "https://example.com/App/?Q=One#Part"
    assert canonicalize_url("https://example.com/app") != canonicalize_url("https://example.com/app/")
    assert canonicalize_url("https://example.com/%7EUser?a=%2F#Frag") == "https://example.com/%7EUser?a=%2F#Frag"
    assert canonicalize_url("https://example.com:443/") == "https://example.com"


def test_canonicalization_is_network_free_and_has_no_brand_allowlist(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    assert canonicalize_url("example.com/") == "https://example.com"
    import src.url_normalization as module
    assert "google" not in inspect.getsource(module).lower()


@pytest.fixture(scope="module")
def service():
    value = URLInference(); value.load_models(); return value


@pytest.mark.parametrize("variants", [
    ["https://www.google.com", "https://www.google.com/", " HTTPS://WWW.GOOGLE.COM/ ", "www.google.com"],
    ["https://github.com", "https://github.com/", "github.com/"],
    ["https://openai.com", "https://openai.com/", "OPENAI.COM"],
])
def test_root_variants_have_identical_predictions(service, variants):
    predictions = [service.predict(value) for value in variants]
    assert len({item["url"] for item in predictions}) == 1
    assert len({item["verdict"] for item in predictions}) == 1
    assert len({item["phishing_probability"] for item in predictions}) == 1


def test_external_suite_schema_and_membership_annotations():
    cases = load_cases(); exact, domains = membership_context()
    assert 70 <= len(cases) <= 120
    assert {case["category"] for case in cases} >= {"known_good", "hard_legitimate", "root_variant", "project_url", "reserved_benign", "synthetic_phishing"}
    for case in cases:
        assert case["expected_label"] in {"LEGITIMATE", "PHISHING"}
        assert isinstance(case["url"], str) and case["url"]
        domain = registered_domain(case["url"])
        assert isinstance(case["url"] in exact, bool)
        assert all(isinstance(domain in domains[split], bool) for split in ("train", "val", "test"))


def test_generated_external_audit_is_complete():
    frame = pd.read_csv("results/external_url_regression.csv")
    assert len(frame) == len(load_cases())
    assert not frame[["normalized_url", "prediction", "model_version", "result"]].isna().any().any()
    assert set(frame.result) <= {"PASS", "FAIL"}
    assert set(frame.exact_url_seen_in_dataset) <= {"YES", "NO"}


def test_context_examples_are_distinguishable():
    benign = analyze_context("https://example.com", "<html><body><h1>Welcome</h1><p>Documentation page.</p></body></html>", "Hi team, The project meeting is tomorrow at 10 AM. Please bring the final report.")
    suspicious = analyze_context("https://example.com", "<html><body><form action='/verify'><input type='text' name='username'><input type='password' name='password'><button>Verify Account</button></form></body></html>", "URGENT: Your account will be suspended. Verify your login immediately and confirm your password.")
    assert suspicious["html"]["password_input_count"] > benign["html"]["password_input_count"]
    assert suspicious["html"]["credential_term_count"] > benign["html"]["credential_term_count"]
    assert suspicious["email"]["risk_term_count"] > benign["email"]["risk_term_count"]
    assert suspicious["context_risk_flags"] and not benign["context_risk_flags"]
    assert suspicious["included_in_validated_probability"] is False


def test_context_endpoint_url_only_and_model_info():
    client = TestClient(app)
    context = client.post("/predict-context", json={"url": "example.com"})
    assert context.status_code == 200
    assert context.json()["context_signals"]["message"] == "No optional context supplied."
    assert context.json()["probabilities_mixed"] is False
    info = client.get("/model-info")
    assert info.status_code == 200 and info.json()["models"]
