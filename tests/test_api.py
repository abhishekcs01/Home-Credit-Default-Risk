from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from src.api.schemas import PredictionItem, PredictResponse


@dataclass
class FakePredictionService:
    model_version: str = "test-double"

    def warmup(self):
        return None

    def predict(self, payload):
        items = [PredictionItem(SK_ID_CURR=rec.SK_ID_CURR, TARGET=0.42) for rec in payload.records]
        return PredictResponse(predictions=items, model_version=self.model_version, request_count=len(items))


def test_health_endpoint(monkeypatch):
    from src.api import main

    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    with TestClient(main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_payload(monkeypatch):
    from src.api import main

    monkeypatch.setattr(main, "get_prediction_service", lambda: FakePredictionService())
    with TestClient(main.app) as client:
        payload = {"records": [{"SK_ID_CURR": 123456, "AMT_INCOME_TOTAL": 125000.0}]}
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] == 1
    assert body["predictions"][0]["SK_ID_CURR"] == 123456
    assert 0.0 <= body["predictions"][0]["TARGET"] <= 1.0
