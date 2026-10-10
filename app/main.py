"""FastAPI application for local phishing detection demo."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .inference import (
    MODEL_VERSION,
    THRESHOLD,
    SELECTED_CNN_WEIGHT,
    SELECTED_LIGHTGBM_WEIGHT,
    get_inference_service,
)
from .inference import URLInference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="AI-Based Phishing Detection System")

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app/static"), name="static")

# Optional research routes stay separate from the validated URL-only endpoint.
from app.research_api import router as research_router

app.include_router(research_router)


from src.laya_detector import analyze_url_with_laya


class URLRequest(BaseModel):
    url: str = Field(min_length=1)


class URLPredictionResponse(BaseModel):
    url: str
    model_version: str
    verdict: str
    phishing_probability: float
    confidence: float
    cnn_probability: float
    lightgbm_probability: float
    reference_ensemble_probability: float
    selected_ensemble_probability: float
    selected_weights: dict[str, float]
    reference_weights: dict[str, float]
    threshold: float
    important_features: dict[str, Any]
    notable_signals: list[str]
    explanation_caveat: str
    bilstm_probability: float | None = None


class DeepAnalysisResponse(BaseModel):
    url: str
    verdict: str
    phishing_probability: float
    confidence: float
    fast_prediction: dict[str, Any]
    laya_analysis: dict[str, Any]
    method: str


@app.get("/health")
async def health() -> dict[str, object]:
    service = get_inference_service()
    try:
        bundle = service.load_models()
    except Exception as exc:
        LOGGER.exception("Model initialization failed during health check")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                "lightgbm_loaded": False,
                "cnn_loaded": False,
            },
        )
    return {
        "status": "ok",
        "lightgbm_loaded": bundle.lightgbm_loaded,
        "cnn_loaded": bundle.cnn_model_loaded,
        "bilstm_loaded": getattr(bundle, "bilstm_loaded", False),
        "device": bundle.device,
        "model_version": "charcnn-bilstm-attention-v3.0" if getattr(bundle, "bilstm_loaded", False) else MODEL_VERSION,
    }


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app/templates/index.html")


@app.post("/predict", response_model=URLPredictionResponse)
async def predict(payload: URLRequest, service: URLInference = Depends(get_inference_service)) -> dict[str, object]:
    try:
        return service.predict(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction model initialization failed")
        raise HTTPException(
            status_code=503,
            detail=f"Model initialization failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/predict-deep", response_model=DeepAnalysisResponse)
async def predict_deep(payload: URLRequest, service: URLInference = Depends(get_inference_service)) -> dict[str, Any]:
    """Tier 2 deep analysis utilizing Convai Innovations' Laya (ModernBERT-large)

    Combines the real-time CharCNN+BiLSTM inference with deep semantic inspection
    and brand impersonation reasoning.
    """
    try:
        fast_result = service.predict(payload.url)
        normalized_url = str(fast_result["url"])
        laya_result = analyze_url_with_laya(normalized_url)

        if laya_result.get("available") and laya_result.get("phishing_probability") is not None:
            laya_prob = float(laya_result["phishing_probability"])
            fast_prob = float(fast_result["phishing_probability"])
            fused_prob = 0.6 * fast_prob + 0.4 * laya_prob
            verdict = "PHISHING" if fused_prob >= THRESHOLD else "LEGITIMATE"
            confidence = fused_prob if verdict == "PHISHING" else (1.0 - fused_prob)
            method = "hybrid_laya_bilstm"
        else:
            verdict = str(fast_result["verdict"])
            fused_prob = float(fast_result["phishing_probability"])
            confidence = float(fast_result["confidence"])
            method = "charcnn_bilstm_edge"

        return {
            "url": normalized_url,
            "verdict": verdict,
            "phishing_probability": fused_prob,
            "confidence": confidence,
            "fast_prediction": fast_result,
            "laya_analysis": laya_result,
            "method": method,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Deep analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"Deep analysis failed: {type(exc).__name__}: {exc}",
        ) from exc


app.state.inference_weights = {
    "selected": "loaded from models/charcnn_bilstm.onnx",
    "threshold": THRESHOLD,
}
