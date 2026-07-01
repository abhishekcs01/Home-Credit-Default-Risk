from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.predict_service import PredictionService, sanitize_probability_series
from src.api.schemas import InferenceRecord, PredictRequest


@dataclass
class FakeEngine:
    calls: list[pd.DataFrame]

    def __init__(self, *_):
        self.calls = []
        self.bundle = {"ready": True}
        self.bundle_loaded = True

    def predict_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(frame.copy())
        # Emit out-of-range/NaN values to verify sanitization.
        raw = np.array([-1.0, np.nan, 2.0, 0.6], dtype=float)
        vals = raw[: len(frame)]
        return pd.DataFrame({"SK_ID_CURR": frame["SK_ID_CURR"].to_numpy(), "TARGET": vals})


def test_sanitize_probability_series_clips_and_fills():
    s = pd.Series([-1.0, np.nan, np.inf, -np.inf, 0.25])
    out = sanitize_probability_series(s)
    assert out.tolist() == [0.0, 0.5, 1.0, 0.0, 0.25]


def test_prediction_service_predict_single_batch(monkeypatch):
    from src.api import predict_service as ps

    monkeypatch.setattr(ps, "EnsembleInferenceEngine", FakeEngine)
    service = PredictionService(model_bundle_path=Path("dummy.pkl"), inference_batch_size=10)
    service.warmup()
    payload = PredictRequest(records=[InferenceRecord(SK_ID_CURR=1), InferenceRecord(SK_ID_CURR=2)])
    response = service.predict(payload)

    assert response.request_count == 2
    assert response.model_version in ("lightgbm_ensemble", "ensemble_v1", "1.0.0")
    assert [p.SK_ID_CURR for p in response.predictions] == [1, 2]
    # From FakeEngine: [-1.0, NaN] -> [0.0, 0.5]
    assert [p.TARGET for p in response.predictions] == [0.0, 0.5]
    assert len(service.engine.calls) == 1


def test_prediction_service_predict_batched(monkeypatch):
    from src.api import predict_service as ps

    monkeypatch.setattr(ps, "EnsembleInferenceEngine", FakeEngine)
    service = PredictionService(model_bundle_path=Path("dummy.pkl"), inference_batch_size=2)
    payload = PredictRequest(
        records=[
            InferenceRecord(SK_ID_CURR=10),
            InferenceRecord(SK_ID_CURR=11),
            InferenceRecord(SK_ID_CURR=12),
        ]
    )
    response = service.predict(payload)
    assert response.request_count == 3
    assert [p.SK_ID_CURR for p in response.predictions] == [10, 11, 12]
    assert len(service.engine.calls) == 2
