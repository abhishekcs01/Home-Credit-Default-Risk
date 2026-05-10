from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.predict import PredictionService
from app.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse
from src.runtime_config import RuntimeSettings, load_runtime_settings

_settings: RuntimeSettings = load_runtime_settings()
_service: PredictionService | None = None


def _configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, _settings.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return logging.getLogger(_settings.logging.logger_name)


logger = _configure_logging()


def _log_event(level: int, event: str, **kwargs: Any) -> None:
    if _settings.logging.structured:
        body = {"event": event, **kwargs}
        logger.log(level, json.dumps(body, default=str))
        return
    logger.log(level, "%s | %s", event, kwargs)


def get_prediction_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService(model_bundle_path=_settings.model.bundle_path)
    return _service


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _settings.api.startup_load_model:
        service = get_prediction_service()
        service.warmup()
        _log_event(logging.INFO, "model_loaded", model_path=str(_settings.model.bundle_path))
    else:
        _log_event(logging.INFO, "startup_skipped_model_load", reason="startup_load_model=false")
    yield


app = FastAPI(
    title=_settings.api.title,
    version=_settings.api.version,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=str(exc), error_type="validation_error").model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception):
    _log_event(logging.ERROR, "unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", error_type="internal_error").model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_service is not None,
        model_path=str(_settings.model.bundle_path),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        service = get_prediction_service()
        response = service.predict(payload)
        _log_event(
            logging.INFO,
            "prediction_success",
            request_count=len(payload.records),
            response_count=response.request_count,
        )
        return response
    except FileNotFoundError as exc:
        _log_event(logging.ERROR, "model_not_found", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        _log_event(logging.ERROR, "prediction_value_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
