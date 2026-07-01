from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass

from fastapi.testclient import TestClient

from src.api.schemas import PredictionItem, PredictResponse
from src.config.settings import ApiSettings, AppConfig


@dataclass
class SlowPredictionService:
    model_version: str = "slow"
    sleep_s: float = 0.0

    class _Engine:
        bundle_loaded = True

    engine = _Engine()

    def warmup(self):
        return None

    def predict(self, payload):
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        items = [PredictionItem(SK_ID_CURR=rec.SK_ID_CURR, TARGET=0.42) for rec in payload.records]
        return PredictResponse(predictions=items, model_version=self.model_version, request_count=len(items))


class FlakyPredictionService(SlowPredictionService):
    def __init__(self):
        self.calls = 0
        super().__init__(model_version="flaky", sleep_s=0.0)

    def predict(self, payload):
        self.calls += 1
        if self.calls == 1:
            time.sleep(1.0)
        return super().predict(payload)


def _api_settings(**overrides):
    from src.api import main

    base = {
        "host": main._cfg.api.host,
        "port": main._cfg.api.port,
        "workers": main._cfg.api.workers,
        "startup_load_model": False,
        "version": main._cfg.api.version,
        "title": main._cfg.api.title,
        "inference_batch_size": main._cfg.api.inference_batch_size,
        "max_request_body_bytes": main._cfg.api.max_request_body_bytes,
        "inference_timeout_seconds": main._cfg.api.inference_timeout_seconds,
        "prediction_retry_count": main._cfg.api.prediction_retry_count,
        "max_concurrent_inferences": main._cfg.api.max_concurrent_inferences,
        "max_pending_requests": main._cfg.api.max_pending_requests,
        "queue_wait_seconds": main._cfg.api.queue_wait_seconds,
        "pending_acquire_seconds": main._cfg.api.pending_acquire_seconds,
        "max_inflight_requests": main._cfg.api.max_inflight_requests,
        "limit_concurrency": main._cfg.api.limit_concurrency,
        "backlog": main._cfg.api.backlog,
    }
    base.update(overrides)
    return ApiSettings(**base)


def test_predict_timeout_returns_fallback(monkeypatch):
    from src.api import main

    cfg = AppConfig(
        api=_api_settings(inference_timeout_seconds=0.5, prediction_retry_count=0),
        training=main._cfg.training,
        load_test=main._cfg.load_test,
    )
    monkeypatch.setattr(main, "_cfg", cfg)
    monkeypatch.setattr(main, "get_prediction_service", lambda: SlowPredictionService(sleep_s=1.0))
    with TestClient(main.app) as client:
        payload = {"records": [{"SK_ID_CURR": 1}]}
        r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["request_count"] == 1
    assert body["predictions"][0]["TARGET"] == 0.5
    assert "timeout_fallback" in body["model_version"]


def test_predict_retries_and_succeeds(monkeypatch):
    from src.api import main

    cfg = AppConfig(
        api=_api_settings(inference_timeout_seconds=0.5, prediction_retry_count=1),
        training=main._cfg.training,
        load_test=main._cfg.load_test,
    )
    flaky = FlakyPredictionService()
    monkeypatch.setattr(main, "_cfg", cfg)
    monkeypatch.setattr(main, "get_prediction_service", lambda: flaky)
    with TestClient(main.app) as client:
        r = client.post("/predict", json={"records": [{"SK_ID_CURR": 3}]})
    assert r.status_code == 200
    assert r.json()["request_count"] == 1
    assert flaky.calls == 2


def test_get_prediction_service_singleton(monkeypatch):
    from src.api import main

    created = {"count": 0}

    class FakeService:
        def __init__(self, *_, **__):
            created["count"] += 1
            self.engine = type("E", (), {"bundle_loaded": True})()

        def warmup(self):
            return None

    monkeypatch.setattr(main, "_service", None)
    monkeypatch.setattr(main, "PredictionService", FakeService)
    first = main.get_prediction_service()
    second = main.get_prediction_service()
    assert first is second
    assert created["count"] == 1


def test_pending_queue_returns_503_when_saturated(monkeypatch):
    from src.api import main

    cfg = AppConfig(
        api=_api_settings(max_pending_requests=1, pending_acquire_seconds=0.05, inference_timeout_seconds=5.0),
        training=main._cfg.training,
        load_test=main._cfg.load_test,
    )
    monkeypatch.setattr(main, "_cfg", cfg)
    monkeypatch.setattr(main, "_service", SlowPredictionService(sleep_s=1.0))
    with TestClient(main.app) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(lambda: client.post("/predict", json={"records": [{"SK_ID_CURR": idx}]}))
                for idx in range(1, 4)
            ]
            results = [future.result() for future in futures]
    statuses = sorted(r.status_code for r in results)
    assert 200 in statuses
    assert 503 in statuses


def test_inference_queue_returns_503_when_saturated(monkeypatch):
    from src.api import main

    cfg = AppConfig(
        api=_api_settings(
            max_concurrent_inferences=1,
            max_pending_requests=8,
            queue_wait_seconds=0.05,
            inference_timeout_seconds=5.0,
        ),
        training=main._cfg.training,
        load_test=main._cfg.load_test,
    )
    monkeypatch.setattr(main, "_cfg", cfg)
    monkeypatch.setattr(main, "_service", SlowPredictionService(sleep_s=1.0))
    with TestClient(main.app) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(
                lambda: client.post("/predict", json={"records": [{"SK_ID_CURR": 1}]}),
            )
            time.sleep(0.05)
            second_future = pool.submit(
                lambda: client.post("/predict", json={"records": [{"SK_ID_CURR": 2}]}),
            )
            first = first_future.result()
            second = second_future.result()
    assert first.status_code == 200
    assert second.status_code == 503
    assert "queue saturated" in second.json()["detail"].lower()


def test_health_bypasses_inflight_capacity(monkeypatch):
    from src.api import main

    cfg = AppConfig(
        api=_api_settings(max_inflight_requests=0),
        training=main._cfg.training,
        load_test=main._cfg.load_test,
    )
    monkeypatch.setattr(main, "_cfg", cfg)
    monkeypatch.setattr(main, "_service", SlowPredictionService())
    with TestClient(main.app) as client:
        r = client.get("/health")
    assert r.status_code == 200
