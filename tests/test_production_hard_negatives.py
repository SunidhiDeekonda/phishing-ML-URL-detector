from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


ROOT = Path(__file__).resolve().parents[1]


def test_hard_negative_suite_is_partitioned_and_large():
    rows = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text())
    assert len(rows) >= 50
    assert {row["hard_negative_split"] for row in rows} == {"train", "validation", "test"}
    assert all(row["expected_label"] == "LEGITIMATE" for row in rows)
    assert all(isinstance(row["exact_url_present_in_original_dataset"], bool) for row in rows)
    assert all(isinstance(row["registered_domain_present_in_training"], bool) for row in rows)


def test_reported_repository_is_held_out_and_legitimate():
    rows = json.loads((ROOT / "tests/data/hard_legitimate_urls.json").read_text())
    target = next(row for row in rows if "SunidhiDeekonda/phishing-ML-URL-detector" in row["url"])
    assert target["hard_negative_split"] == "test"
    response = TestClient(app).post("/predict", json={"url": target["url"]})
    assert response.status_code == 200
    assert response.json()["verdict"] == "LEGITIMATE"


def test_github_repository_regression_examples_are_legitimate():
    client = TestClient(app)
    for url in [
        "https://github.com/openai/openai-python",
        "https://github.com/microsoft/vscode",
        "https://github.com/vercel/next.js",
        "https://github.com/pallets/flask",
        "https://github.com/python/cpython",
    ]:
        assert client.post("/predict", json={"url": url}).json()["verdict"] == "LEGITIMATE"


def test_frontend_never_directly_renders_object_objects():
    script = (ROOT / "app/static/app.js").read_text()
    assert "[object Object]" not in script
    assert "signalMarkup(payload.signals" in script
    assert "riskMarkup(payload.context_risk)" in script


def test_phishing_response_has_honest_explainability_caveat():
    response = TestClient(app).post("/predict", json={"url": "http://secure-account-login-example.xyz/verify"})
    payload = response.json()
    assert payload["verdict"] == "PHISHING"
    assert payload["notable_signals"]
    assert "false positives" in payload["explanation_caveat"]
