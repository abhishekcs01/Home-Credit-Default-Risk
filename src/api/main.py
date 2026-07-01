from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src import config
from src.api.predict_service import PredictionService
from src.api.schemas import ErrorResponse, HealthResponse, PredictionItem, PredictRequest, PredictResponse
from src.config import load_config
from src.utils import get_logger, init_logging

_cfg = load_config()
_service: PredictionService | None = None
_service_lock = threading.Lock()
_inflight_lock = threading.Lock()
_inflight_active = 0
_inference_executor: ThreadPoolExecutor | None = None
_inference_slots: asyncio.Semaphore | None = None
_pending_slots: asyncio.Semaphore | None = None
LOGGER = get_logger("api.main")


def get_prediction_service() -> PredictionService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = PredictionService(
                    model_bundle_path=config.MODEL_BUNDLE_PATH,
                    inference_batch_size=_cfg.api.inference_batch_size,
                )
    return _service


def _get_inference_executor() -> ThreadPoolExecutor:
    global _inference_executor
    if _inference_executor is None:
        _inference_executor = ThreadPoolExecutor(
            max_workers=_cfg.api.max_concurrent_inferences,
            thread_name_prefix="inference",
        )
    return _inference_executor


def _is_predict_path(path: str) -> bool:
    return path in {"/predict", "/v1/predict"}


def _is_health_path(path: str) -> bool:
    return path in {"/health", "/v1/health"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _inference_slots, _inference_executor, _pending_slots
    init_logging()
    LOGGER.info("API startup requested")
    _inference_slots = asyncio.Semaphore(_cfg.api.max_concurrent_inferences)
    _pending_slots = asyncio.Semaphore(_cfg.api.max_pending_requests)
    _inference_executor = ThreadPoolExecutor(
        max_workers=_cfg.api.max_concurrent_inferences,
        thread_name_prefix="inference",
    )
    if _cfg.api.startup_load_model:
        get_prediction_service().warmup()
    LOGGER.info(
        "API startup completed (max_concurrent_inferences=%d, max_pending_requests=%d, queue_wait_seconds=%.1f)",
        _cfg.api.max_concurrent_inferences,
        _cfg.api.max_pending_requests,
        _cfg.api.queue_wait_seconds,
    )
    yield
    _inference_executor.shutdown(wait=False, cancel_futures=True)
    _inference_executor = None


app = FastAPI(title=_cfg.api.title, version=_cfg.api.version, lifespan=lifespan)


@app.middleware("http")
async def inflight_protection_middleware(request, call_next):
    global _inflight_active
    if _is_health_path(request.url.path):
        return await call_next(request)
    with _inflight_lock:
        if _inflight_active >= _cfg.api.max_inflight_requests:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    detail="Server busy. Retry shortly.",
                    error_type="capacity_exceeded",
                ).model_dump(),
            )
        _inflight_active += 1
    try:
        return await call_next(request)
    finally:
        with _inflight_lock:
            _inflight_active = max(0, _inflight_active - 1)


@app.get("/health", response_model=HealthResponse, tags=["health"])
@app.get("/v1/health", response_model=HealthResponse, tags=["v1"])
def health() -> HealthResponse:
    bundle_path = config.MODEL_BUNDLE_PATH
    bundle_exists = bundle_path.is_file()
    service = _service
    bundle_loaded = bool(service and service.engine.bundle_loaded)
    return HealthResponse(
        status="ok",
        model_loaded=service is not None,
        model_path=str(bundle_path),
        inference_ready=bundle_loaded and bundle_exists,
        bundle_path_exists=bundle_exists,
    )


async def _run_predict(payload: PredictRequest) -> PredictResponse:
    if len(payload.records) > _cfg.api.max_batch_size:
        raise HTTPException(status_code=422, detail=f"Batch size exceeds max_batch_size={_cfg.api.max_batch_size}")

    pending = _pending_slots
    slots = _inference_slots
    if pending is None or slots is None:
        raise HTTPException(status_code=503, detail="Inference queue is not ready")

    pending_wait_s = max(0.1, float(_cfg.api.pending_acquire_seconds))
    queue_wait_s = max(0.0, float(_cfg.api.queue_wait_seconds))

    pending_acquired = False
    slot_acquired = False
    try:
        await asyncio.wait_for(pending.acquire(), timeout=pending_wait_s)
        pending_acquired = True
        try:
            await asyncio.wait_for(slots.acquire(), timeout=queue_wait_s if queue_wait_s > 0 else None)
            slot_acquired = True
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="Inference queue saturated. Retry shortly.",
                headers={"Retry-After": "10"},
            ) from exc

        timeout_s = max(0.5, float(_cfg.api.inference_timeout_seconds))
        retries = max(0, int(_cfg.api.prediction_retry_count))
        service = get_prediction_service()
        loop = asyncio.get_running_loop()
        executor = _get_inference_executor()

        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(executor, service.predict, payload),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                if attempt == retries:
                    LOGGER.exception("Inference timed out after %d attempts; returning fallback response", retries + 1)
                    fallback_items = [PredictionItem(SK_ID_CURR=rec.SK_ID_CURR, TARGET=0.5) for rec in payload.records]
                    return PredictResponse(
                        predictions=fallback_items,
                        model_version=f"{service.model_version}_timeout_fallback",
                        request_count=len(fallback_items),
                    )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="Too many requests waiting. Retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc
    finally:
        if slot_acquired:
            slots.release()
        if pending_acquired:
            pending.release()

    raise HTTPException(status_code=500, detail="Inference failed")  # pragma: no cover


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["legacy"],
)
async def predict(payload: PredictRequest) -> PredictResponse:
    return await _run_predict(payload)


@app.post(
    "/v1/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["v1"],
)
async def predict_v1(payload: PredictRequest) -> PredictResponse:
    return await _run_predict(payload)
