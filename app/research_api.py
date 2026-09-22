"""Optional research API routes; all analysis remains network-free."""

from __future__ import annotations
import json
import os
from functools import lru_cache
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.inference import URLInference
from src.context_features import analyze_context
from src.continuous_learning import FeedbackStore

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter(tags=["research-extensions"])


class ContextRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    html: str | None = Field(default=None, max_length=200_000)
    email_text: str | None = Field(default=None, max_length=200_000)


class FeedbackRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    predicted_label: str | int
    correct_label: str | int
    optional_notes: str = Field(default="", max_length=500)


@lru_cache(maxsize=1)
def _inference() -> URLInference:
    service = URLInference(); service.load_models(); return service


@lru_cache(maxsize=1)
def _feedback_store() -> tuple[FeedbackStore, bool]:
    hash_path = ROOT / "models/test_url_hashes.json"
    test_hashes = json.loads(hash_path.read_text()) if hash_path.exists() else []
    ephemeral = bool(os.getenv("VERCEL"))
    default = Path("/tmp/phishing_feedback.jsonl") if ephemeral else ROOT / "data/feedback/pending.jsonl"
    return FeedbackStore(Path(os.getenv("FEEDBACK_STORE_PATH", str(default))), test_hashes=test_hashes), not ephemeral


@router.post("/predict-context")
def predict_context(request: ContextRequest) -> dict[str, object]:
    try:
        prediction = _inference().predict(request.url)
        return {"validated_url_prediction": prediction, "context_signals": analyze_context(request.url, request.html, request.email_text), "probabilities_mixed": False}
    except Exception as exc: raise HTTPException(500, f"Context analysis failed: {type(exc).__name__}") from exc


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest) -> dict[str, object]:
    try:
        store, persistent = _feedback_store()
        return {**store.submit(request.url, request.predicted_label, request.correct_label, request.optional_notes), "persistent_storage": persistent, "automatic_retraining": False}
    except ValueError as exc: raise HTTPException(400, str(exc)) from exc


@router.get("/model-info")
def model_info() -> dict[str, object]:
    path = ROOT / "models/model_registry.json"
    return json.loads(path.read_text()) if path.exists() else {"models": []}
