from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.schemas import PredictRequest, PredictResponse, PredictionItem
from src.inference import EnsembleInferenceEngine


@dataclass
class PredictionService:
    model_bundle_path: Path
    model_version: str = "lightgbm_ensemble"

    def __post_init__(self) -> None:
        self.engine = EnsembleInferenceEngine(self.model_bundle_path)

    def warmup(self) -> None:
        _ = self.engine.bundle

    def predict(self, payload: PredictRequest) -> PredictResponse:
        rows = [item.to_feature_dict() for item in payload.records]
        inference_frame = pd.DataFrame(rows)
        predictions = self.engine.predict_dataframe(inference_frame)
        items = [
            PredictionItem(SK_ID_CURR=int(rec["SK_ID_CURR"]), TARGET=float(rec["TARGET"]))
            for rec in predictions.to_dict(orient="records")
        ]
        return PredictResponse(
            predictions=items,
            model_version=self.model_version,
            request_count=len(items),
        )
