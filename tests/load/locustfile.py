from __future__ import annotations

import json
import os
from pathlib import Path

from locust import HttpUser, between, task

from src.runtime_config import load_runtime_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = PROJECT_ROOT / "examples" / "payloads"

_settings = load_runtime_settings()
_min_wait = float(os.getenv("HOME_CREDIT_LOAD_MIN_WAIT_SECONDS", _settings.load_test.min_wait_seconds))
_max_wait = float(os.getenv("HOME_CREDIT_LOAD_MAX_WAIT_SECONDS", _settings.load_test.max_wait_seconds))
_connect_timeout = float(os.getenv("HOME_CREDIT_LOAD_CONNECT_TIMEOUT_SECONDS", "10"))
_read_timeout = float(os.getenv("HOME_CREDIT_LOAD_READ_TIMEOUT_SECONDS", "180"))
_REQUEST_TIMEOUT = (_connect_timeout, _read_timeout)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


MINIMAL_PAYLOAD = _load_json(PAYLOAD_DIR / "minimal_request.json")
FULL_PAYLOAD = _load_json(PAYLOAD_DIR / "full_application_request.json")
BATCH_PAYLOAD = _load_json(PAYLOAD_DIR / "batch_request.json")


class HomeCreditInferenceUser(HttpUser):
    host = os.getenv("HOME_CREDIT_LOAD_HOST", _settings.load_test.host)
    wait_time = between(_min_wait, _max_wait)

    @task(5)
    def predict_minimal(self):
        with self.client.post(
            "/predict",
            json=MINIMAL_PAYLOAD,
            name="predict_minimal",
            catch_response=True,
            timeout=_REQUEST_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}, body={resp.text}")

    @task(2)
    def predict_full(self):
        with self.client.post(
            "/predict",
            json=FULL_PAYLOAD,
            name="predict_full",
            catch_response=True,
            timeout=_REQUEST_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}, body={resp.text}")

    @task(1)
    def predict_batch(self):
        with self.client.post(
            "/predict",
            json=BATCH_PAYLOAD,
            name="predict_batch",
            catch_response=True,
            timeout=_REQUEST_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}, body={resp.text}")
