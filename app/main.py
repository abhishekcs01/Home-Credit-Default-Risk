from __future__ import annotations

import json
import logging
import threading
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.predict import PredictionService
from app.schemas import ErrorResponse, HealthResponse, PredictRequest, PredictResponse
from src.runtime_config import RuntimeSettings, load_runtime_settings

_settings: RuntimeSettings = load_runtime_settings()
_service: PredictionService | None = None
_service_lock = threading.Lock()


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
        with _service_lock:
            if _service is None:
                _service = PredictionService(model_bundle_path=_settings.model.bundle_path)
    return _service


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        from pandas.errors import PerformanceWarning
    except ImportError:
        PerformanceWarning = None  # type: ignore[misc, assignment]
    if PerformanceWarning is not None:
        warnings.filterwarnings("ignore", category=PerformanceWarning)

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


@app.middleware("http")
async def enforce_max_request_body(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit():
            n = int(cl)
            if n > _settings.api.max_request_body_bytes:
                _log_event(
                    logging.WARNING,
                    "payload_too_large",
                    content_length=n,
                    limit=_settings.api.max_request_body_bytes,
                    path=str(request.url.path),
                )
                return JSONResponse(
                    status_code=413,
                    content=ErrorResponse(
                        detail=f"Request body exceeds limit of {_settings.api.max_request_body_bytes} bytes",
                        error_type="payload_too_large",
                    ).model_dump(),
                )
    return await call_next(request)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    path = request.url.path
    if path not in ("/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"):
        _log_event(
            logging.INFO,
            "http_request",
            method=request.method,
            path=path,
            duration_ms=round(elapsed_ms, 3),
            status_code=response.status_code,
        )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    _log_event(
        logging.WARNING,
        "request_validation_failed",
        path=str(request.url.path),
        errors=errors,
    )
    detail_parts: list[str] = []
    for e in errors:
        loc = ".".join(str(x) for x in e.get("loc", ()))
        detail_parts.append(f"{loc}: {e.get('msg')}")
    detail = "; ".join(detail_parts) if detail_parts else str(exc)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=detail, error_type="validation_error").model_dump(),
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(request: Request, exc: ResponseValidationError):
    logger.exception("response_validation_failed path=%s", request.url.path)
    _log_event(logging.ERROR, "response_validation_failed", path=str(request.url.path), errors=exc.errors())
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", error_type="response_validation_error").model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Let FastAPI/Starlette handle intentional HTTP errors (e.g. raise HTTPException from /predict).
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception("unhandled_exception path=%s", request.url.path)
    _log_event(logging.ERROR, "unhandled_exception", path=str(request.url.path), error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", error_type="internal_error").model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle_path = _settings.model.bundle_path
    bundle_exists = bundle_path.is_file()
    svc = _service
    bundle_loaded = False
    if svc is not None:
        eng = getattr(svc, "engine", None)
        if eng is not None:
            bundle_loaded = bool(getattr(eng, "bundle_loaded", False))
    inference_ready = bundle_loaded and bundle_exists
    return HealthResponse(
        status="ok",
        model_loaded=svc is not None,
        model_path=str(bundle_path),
        inference_ready=inference_ready,
        bundle_path_exists=bundle_exists,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
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
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        _log_event(logging.ERROR, "model_not_found", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        _log_event(logging.ERROR, "prediction_value_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("prediction_failed")
        _log_event(logging.ERROR, "prediction_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Inference failed") from exc
