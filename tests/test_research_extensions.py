import socket
import urllib.request
import numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient
from app.main import app
from src.adversarial_urls import MUTATION_TYPES, mutate_url
from src.context_features import analyze_context, analyze_email, analyze_html
from src.continuous_learning import FeedbackStore, ModelRegistry, validation_gate
from src.drift_monitor import create_drift_report


def test_adversarial_mutations_are_deterministic_and_offline(monkeypatch):
    monkeypatch.setattr(socket,"create_connection",lambda *a,**k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    source="http://secure-account.example/login?verify=1"
    for kind in MUTATION_TYPES:
        first=mutate_url(source,kind,source_row_index=17)
        assert first==mutate_url(source,kind,source_row_index=17) and first!=source


def test_adversarial_suites_preserve_split_integrity():
    data=pd.read_csv("data/processed/dataset.csv"); train=set(np.load("data/processed/train_idx.npy")); val=set(np.load("data/processed/val_idx.npy")); test=set(np.load("data/processed/test_idx.npy")); av=pd.read_csv("data/processed/adversarial_validation.csv"); at=pd.read_csv("data/processed/adversarial_test.csv")
    assert set(av.source_row_index)<=val and set(at.source_row_index)<=test and not set(at.source_row_index)&train
    assert len(av)==len(val) and len(at)==len(test) and set(at.label)=={0,1} and len(data)==20_000


def test_context_extraction_is_separate_from_probability():
    result=analyze_context("https://example.com/login",'<form action="https://outside.example/collect"><input type="password"><meta http-equiv="refresh"></form>',"Urgent: verify your account. Click here and enter your password https://example.test")
    assert result["html"]["password_input_count"]==1 and result["html"]["external_target_count"]==1 and result["email"]["url_count"]==1
    assert result["context_risk_flags"] and result["included_in_validated_probability"] is False and result["automatic_url_fetching"] is False


def test_feedback_validation_deduplication_and_test_protection(tmp_path):
    store=FeedbackStore(tmp_path/"feedback.jsonl",{"https://heldout.example/test"}); first=store.submit("https://new.example/login","PHISHING","LEGITIMATE","review"); second=store.submit("https://new.example/login",1,0)
    assert first["deduplicated"] is False and second["deduplicated"] is True and len(store.read_all())==1
    with pytest.raises(ValueError,match="Held-out"): store.submit("https://heldout.example/test",1,0)
    with pytest.raises(ValueError,match="Label"): store.submit("https://other.example","MAYBE",0)


def test_model_registry_and_drift_report(tmp_path):
    registry=ModelRegistry(tmp_path/"registry.json"); registry.register({"version":"test-1","training_date":"2026-09-22","data_count":10,"clean_validation_metrics":{"roc_auc":.9},"robustness_validation_metrics":{"roc_auc":.8},"status":"candidate"})
    assert registry.load()["models"][0]["version"]=="test-1" and create_drift_report(pd.DataFrame({"x":np.arange(100)}),[])["status"]=="insufficient_data"


def test_human_review_candidate_dataset_and_validation_gate(tmp_path):
    base=tmp_path/"base.csv"; pd.DataFrame({"url":["https://base.example"],"label":[0]}).to_csv(base,index=False)
    store=FeedbackStore(tmp_path/"feedback.jsonl"); submitted=store.submit("https://approved.example/login",1,1)
    store.review(submitted["record_id"],"approved"); output=tmp_path/"candidate.csv"
    assert store.build_candidate_dataset(base,output)==1 and len(pd.read_csv(output))==2
    assert validation_gate(.99,.90,.989,.93)["accepted"] is True
    assert validation_gate(.99,.90,.98,.99)["accepted"] is False


def test_robust_model_artifacts_load():
    import joblib, torch
    bundle=joblib.load("models/lightgbm_robust.pkl"); checkpoint=torch.load("models/char_cnn_robust.pt",map_location="cpu",weights_only=False)
    assert {"model","calibrator","feature_columns"}<=set(bundle) and "model_state_dict" in checkpoint


def test_predict_context_endpoint(monkeypatch):
    class FakeInference:
        def predict(self,url): return {"url":url,"verdict":"LEGITIMATE","phishing_probability":.1}
    import app.research_api as research_api
    monkeypatch.setattr(research_api,"_inference",lambda:FakeInference())
    response=TestClient(app).post("/predict-context",json={"url":"https://example.com","html":"<form><input type='password'></form>","email_text":"verify account"})
    assert response.status_code==200 and response.json()["probabilities_mixed"] is False and response.json()["context_signals"]["html"]["form_count"]==1


SAFE_HTML = "<html><body><h1>Welcome</h1><p>This is a documentation page.</p></body></html>"
SUSPICIOUS_HTML = "<html><body><form action='/verify'><input type='text' name='username'><input type='password' name='password'><button>Verify Account</button></form></body></html>"
SAFE_EMAIL = "Hi team,\n\nThe project meeting is tomorrow at 10 AM.\nPlease bring the final report.\n\nThanks."
SUSPICIOUS_EMAIL = "URGENT: Your account will be suspended.\n\nVerify your login immediately and confirm your password to prevent account closure."


def test_html_endpoint_is_independent_and_distinguishes_examples():
    client = TestClient(app)
    safe = client.post("/analyze-html", json={"html": SAFE_HTML})
    suspicious = client.post("/analyze-html", json={"html": SUSPICIOUS_HTML})
    assert safe.status_code == suspicious.status_code == 200
    assert safe.json()["analysis_type"] == "html"
    assert "url" not in safe.json() and "email" not in safe.json()
    assert suspicious.json()["signals"]["form_count"] == 1
    assert suspicious.json()["signals"]["password_input_count"] == 1
    assert suspicious.json()["signals"]["credential_term_count"] > safe.json()["signals"]["credential_term_count"]
    assert suspicious.json()["context_risk"]["level"] == "ELEVATED"
    assert suspicious.json()["network_access"] is False and suspicious.json()["html_executed"] is False
    assert suspicious.json()["included_in_validated_probability"] is False


def test_email_endpoint_is_independent_and_distinguishes_examples():
    client = TestClient(app)
    safe = client.post("/analyze-email", json={"email_text": SAFE_EMAIL})
    suspicious = client.post("/analyze-email", json={"email_text": SUSPICIOUS_EMAIL})
    assert safe.status_code == suspicious.status_code == 200
    assert safe.json()["analysis_type"] == "email"
    assert "url" not in safe.json() and "html" not in safe.json()
    assert suspicious.json()["signals"]["risk_term_count"] > safe.json()["signals"]["risk_term_count"]
    assert suspicious.json()["signals"]["credential_request_count"] == 1
    assert suspicious.json()["context_risk"]["level"] == "ELEVATED"
    assert suspicious.json()["network_access"] is False and suspicious.json()["mailbox_access"] is False
    assert suspicious.json()["included_in_validated_probability"] is False


@pytest.mark.parametrize(("path", "payload"), [
    ("/analyze-html", {"html": "   "}),
    ("/analyze-email", {"email_text": "\n\t"}),
])
def test_independent_context_endpoints_reject_empty_text(path, payload):
    response = TestClient(app).post(path, json=payload)
    assert response.status_code == 400
    assert "Paste" in response.json()["detail"]


def test_independent_context_analysis_is_network_free(monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("network access forbidden")
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    assert analyze_html(SUSPICIOUS_HTML)["network_access"] is False
    assert analyze_email(SUSPICIOUS_EMAIL)["network_access"] is False


def test_frontend_exposes_three_independent_tools():
    page = TestClient(app).get("/").text
    assert "URL Phishing Detector" in page
    assert 'id="analyseHtmlBtn"' in page and 'id="htmlInput"' in page and 'id="htmlResult"' in page
    assert 'id="analyseEmailBtn"' in page and 'id="emailInput"' in page and 'id="emailResult"' in page
    assert "HTML Phishing Context Analyzer" in page
    assert "Email Phishing Context Analyzer" in page
