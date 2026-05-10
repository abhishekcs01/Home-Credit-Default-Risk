from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas import PredictRequest, PredictResponse, PredictionItem
from src.inference import EnsembleInferenceEngine

logger = logging.getLogger(__name__)


def sanitize_probability_series(series: pd.Series) -> pd.Series:
    """Ensure finite probabilities in [0, 1] for Pydantic response validation and stable JSON."""
    arr = np.asarray(series, dtype=np.float64)
    if not np.isfinite(arr).all():
        n_bad = int(np.sum(~np.isfinite(arr)))
        logger.warning("Sanitizing %s non-finite probability value(s) for API output", n_bad)
        arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    clipped = np.clip(arr, 0.0, 1.0)
    return pd.Series(clipped, index=series.index, dtype=float)


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
        predictions = predictions.copy()
        predictions["TARGET"] = sanitize_probability_series(predictions["TARGET"])
        items = [
            PredictionItem(SK_ID_CURR=int(rec["SK_ID_CURR"]), TARGET=float(rec["TARGET"]))
            for rec in predictions.to_dict(orient="records")
        ]
        return PredictResponse(
            predictions=items,
            model_version=self.model_version,
            request_count=len(items),
        )
