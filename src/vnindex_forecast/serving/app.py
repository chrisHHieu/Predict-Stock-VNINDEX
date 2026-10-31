"""FastAPI service for next-session VNINDEX direction.

The client posts the last bars it holds (at least 60 daily OHLCV rows). The
server rebuilds the same features used in training and returns the probability
of an up session, the predicted log return and the current model version.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vnindex_forecast import __version__
from vnindex_forecast.config import AppConfig, load_config
from vnindex_forecast.features.build import build_feature_frame_from_bars
from vnindex_forecast.models import registry


class Bar(BaseModel):
    date: str | None = None
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class PredictRequest(BaseModel):
    bars: list[Bar] = Field(min_length=2, description="Daily OHLCV bars, oldest first")


class BatchRequest(BaseModel):
    windows: list[list[Bar]] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    as_of: str
    probability_up: float
    predicted_next_return: float
    decision: str
    threshold: float
    model_version: str
    bars_used: int


class BatchResponse(BaseModel):
    predictions: list[Prediction]


def _predict_one(artifact: registry.Artifact, config: AppConfig, bars: list[Bar]) -> Prediction:
    payload = artifact.payload
    names: list[str] = list(payload["feature_names"])
    frame = build_feature_frame_from_bars([bar.model_dump() for bar in bars], config)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"feature mismatch with the artifact: {missing}")

    latest = frame.iloc[[-1]]
    if latest[names].isna().any().any():
        raise ValueError(
            f"features are not complete after {len(bars)} bars; "
            f"send at least {config.features.min_history} bars"
        )

    probability = float(payload["classifier"].predict_proba(latest[names])[0, 1])
    predicted_return = float(payload["regressor"].predict(latest[names])[0])
    threshold = float(payload["threshold"])
    return Prediction(
        as_of=latest["date"].iloc[0].date().isoformat(),
        probability_up=round(probability, 6),
        predicted_next_return=round(predicted_return, 6),
        decision="long" if probability > threshold else "flat",
        threshold=threshold,
        model_version=artifact.version,
        bars_used=len(bars),
    )


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Build the application. Tests pass a config with temporary paths."""
    cfg = config or load_config(os.environ.get("VNINDEX_CONFIG"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.artifact = registry.load_artifact(cfg)
        except FileNotFoundError:
            application.state.artifact = None
        yield

    application = FastAPI(
        title="VNINDEX next-session forecast",
        description=(
            "Walk-forward trained LightGBM direction model, "
            "served from the artifact registry."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.state.config = cfg
    application.state.artifact = None

    def _artifact() -> registry.Artifact:
        artifact = application.state.artifact
        if artifact is None:
            raise HTTPException(
                status_code=503,
                detail="no model artifact loaded; run `python -m vnindex_forecast train`",
            )
        return artifact

    @application.get("/health")
    def health() -> dict[str, Any]:
        artifact = application.state.artifact
        return {
            "status": "ok" if artifact is not None else "degraded",
            "model_loaded": artifact is not None,
            "model_version": artifact.version if artifact is not None else None,
        }

    @application.get("/model/info")
    def model_info() -> dict[str, Any]:
        info = _artifact().as_info()
        info["feature_count"] = len(info.get("features") or [])
        return info

    @application.post("/predict", response_model=Prediction)
    def predict(request: PredictRequest) -> Prediction:
        artifact = _artifact()
        try:
            return _predict_one(artifact, cfg, request.bars)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(request: BatchRequest) -> BatchResponse:
        artifact = _artifact()
        predictions = []
        for index, window in enumerate(request.windows):
            try:
                predictions.append(_predict_one(artifact, cfg, window))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=f"window {index}: {exc}") from exc
        return BatchResponse(predictions=predictions)

    return application


app = create_app()
