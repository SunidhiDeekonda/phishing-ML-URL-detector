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


class URLRequest(BaseModel):
    url: str = Field(min_length=1)


class URLPredictionResponse(BaseModel):
    url: str
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
        "device": bundle.device,
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


app.state.inference_weights = {
    "selected": {
        "cnn": SELECTED_CNN_WEIGHT,
        "lightgbm": SELECTED_LIGHTGBM_WEIGHT,
    },
    "threshold": THRESHOLD,
}
