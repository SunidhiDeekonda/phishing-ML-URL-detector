"""FastAPI application for local phishing detection demo."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .inference import (
    THRESHOLD,
    SELECTED_CNN_WEIGHT,
    SELECTED_LIGHTGBM_WEIGHT,
    get_inference_service,
)
from .inference import URLInference

app = FastAPI(title="AI-Based Phishing Detection System")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


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


@app.on_event("startup")
async def start_services() -> None:
    service = get_inference_service()
    service.load_models()


@app.get("/health")
async def health() -> dict[str, object]:
    service = get_inference_service()
    bundle = service._bundle
    return {
        "status": "ok",
        "lightgbm_loaded": bool(bundle and bundle.lightgbm_loaded),
        "cnn_loaded": bool(bundle and bundle.cnn_model_loaded),
        "device": bundle.device if bundle else "uninitialized",
    }


@app.get("/")
async def home() -> FileResponse:
    return FileResponse("app/templates/index.html")


@app.post("/predict", response_model=URLPredictionResponse)
async def predict(payload: URLRequest, service: URLInference = Depends(get_inference_service)) -> dict[str, object]:
    try:
        return service.predict(payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.state.inference_weights = {
    "selected": {
        "cnn": SELECTED_CNN_WEIGHT,
        "lightgbm": SELECTED_LIGHTGBM_WEIGHT,
    },
    "threshold": THRESHOLD,
}
