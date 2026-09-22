import socket
import numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient
from app.main import app
from src.adversarial_urls import MUTATION_TYPES, mutate_url
from src.context_features import analyze_context
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
