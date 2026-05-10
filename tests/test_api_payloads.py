from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.schemas import PredictResponse, PredictionItem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAYLOADS_DIR = PROJECT_ROOT / "examples" / "payloads"


@dataclass
class FakePredictionService:
    model_version: str = "test-double"

    def predict(self, payload):
        items = [PredictionItem(SK_ID_CURR=rec.SK_ID_CURR, TARGET=0.51) for rec in payload.records]
        return PredictResponse(predictions=items, model_version=self.model_version, request_count=len(items))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HOME_CREDIT_API_STARTUP_LOAD_MODEL", "false")
    from app import main

    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.mark.parametrize(
    "payload_file",
    [
        "minimal_request.json",
        "full_application_request.json",
        "batch_request.json",
    ],
)
def test_example_payloads_roundtrip(client, payload_file):
    payload_path = PAYLOADS_DIR / payload_file
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert body["request_count"] == len(payload["records"])


def test_malformed_payload_is_rejected(client):
    response = client.post(
        "/predict",
        content='{"records":[{"SK_ID_CURR": 1}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["error_type"] == "validation_error"


def test_response_schema_shape(client):
    payload = {"records": [{"SK_ID_CURR": 424242, "AMT_CREDIT": 350000.0}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"predictions", "model_version", "request_count"}
    assert set(body["predictions"][0].keys()) == {"SK_ID_CURR", "TARGET"}
