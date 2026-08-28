from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request

import pytest


@pytest.fixture(scope="session")
def running_app_server() -> str:
    command = [
        ".venv/bin/python",
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8005",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        started = False
        for _ in range(80):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8005/health", timeout=1) as response:
                    if response.status == 200:
                        started = True
                        break
            except Exception:
                time.sleep(0.15)
        if not started:
            raise RuntimeError("Demo server failed to start")
        yield "http://127.0.0.1:8005"
    finally:
        process.terminate()
        process.wait(timeout=10)


def _post_json(url: str, payload: dict[str, object]) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return exc.code, body


def _get_json(url: str, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"{url}{path}", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body


def test_health_endpoint_works(running_app_server: str) -> None:
    status, payload = _get_json(running_app_server, "/health")
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["lightgbm_loaded"] is True
    assert payload["cnn_loaded"] is True
    assert payload["device"] in {"mps", "cuda", "cpu"}


@pytest.mark.parametrize(
    "url",
    [
        "https://www.google.com",
        "https://example.com",
        "https://accounts.google.com",
        "http://secure-account-login-example.xyz/verify",
    ],
)
def test_predict_endpoint_returns_probs(running_app_server: str, url: str) -> None:
    status, payload = _post_json(running_app_server, {"url": url})
    assert status == 200
    assert payload["url"] == url.lower()
    assert payload["verdict"] in {"PHISHING", "LEGITIMATE"}
    assert 0.0 <= payload["cnn_probability"] <= 1.0
    assert 0.0 <= payload["lightgbm_probability"] <= 1.0
    assert 0.0 <= payload["reference_ensemble_probability"] <= 1.0
    assert 0.0 <= payload["selected_ensemble_probability"] <= 1.0
    assert payload["important_features"]


def test_predict_invalid_input_is_rejected(running_app_server: str) -> None:
    status, _payload = _post_json(running_app_server, {"url": ""})
    assert status in {400, 422}
