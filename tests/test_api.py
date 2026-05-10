from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.schemas import PredictResponse, PredictionItem


@dataclass
class FakePredictionService:
    model_version: str = "test-double"

    def predict(self, payload):
        items = [PredictionItem(SK_ID_CURR=rec.SK_ID_CURR, TARGET=0.42) for rec in payload.records]
        return PredictResponse(predictions=items, model_version=self.model_version, request_count=len(items))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("HOME_CREDIT_API_STARTUP_LOAD_MODEL", "false")
    from app import main

    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "model_path" in body


def test_predict_valid_payload(client):
    payload = {
        "records": [
            {"SK_ID_CURR": 123456, "AMT_INCOME_TOTAL": 125000.0, "ORGANIZATION_TYPE": "Business Entity Type 3"}
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] == 1
    assert body["predictions"][0]["SK_ID_CURR"] == 123456
    assert 0.0 <= body["predictions"][0]["TARGET"] <= 1.0


def test_predict_invalid_missing_required_field(client):
    payload = {"records": [{"AMT_INCOME_TOTAL": 110000.0}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error_type"] == "validation_error"


def test_predict_invalid_datatype(client):
    payload = {"records": [{"SK_ID_CURR": "not-an-int", "AMT_INCOME_TOTAL": 110000.0}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error_type"] == "validation_error"


def test_batch_prediction(client):
    payload = {
        "records": [
            {"SK_ID_CURR": 1, "AMT_INCOME_TOTAL": 100000.0},
            {"SK_ID_CURR": 2, "AMT_INCOME_TOTAL": 200000.0},
            {"SK_ID_CURR": 3, "AMT_INCOME_TOTAL": 300000.0},
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] == 3
    assert len(body["predictions"]) == 3
