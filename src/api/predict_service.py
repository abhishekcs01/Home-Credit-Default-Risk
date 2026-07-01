from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.schemas import PredictionItem, PredictRequest, PredictResponse
from src.models.predict import EnsembleInferenceEngine
from src.utils import get_logger

LOGGER = get_logger("api.predict_service")


def sanitize_probability_series(series: pd.Series) -> pd.Series:
    arr = np.asarray(series, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    clipped = np.clip(arr, 0.0, 1.0)
    return pd.Series(clipped, index=series.index, dtype=float)


@dataclass
class PredictionService:
    model_bundle_path: Path
    model_version: str | None = None
    inference_batch_size: int = 512

    def __post_init__(self) -> None:
        self.engine = EnsembleInferenceEngine(self.model_bundle_path)
        if self.model_version is None:
            bundle = self.engine.bundle
            self.model_version = str(bundle.get("version") or bundle.get("metrics", {}).get("version") or "ensemble_v1")

    def warmup(self) -> None:
        LOGGER.info("Warming up model bundle from %s", self.model_bundle_path)
        _ = self.engine.bundle

    def predict(self, payload: PredictRequest) -> PredictResponse:
        rows = [item.to_feature_dict() for item in payload.records]
        inference_frame = pd.DataFrame(rows)
        expected = list(self.engine.bundle.get("feature_names") or [])
        if expected:
            missing = [c for c in expected if c not in inference_frame.columns and c != "SK_ID_CURR"]
            if missing:
                LOGGER.warning(
                    "Inference frame missing %d of %d model features; imputation will apply.",
                    len(missing),
                    len(expected),
                )
        LOGGER.info("Received prediction request with %d records", len(inference_frame))
        predictions = self._predict_batched(inference_frame)
        predictions = predictions.copy()
        predictions["TARGET"] = sanitize_probability_series(predictions["TARGET"])
        items = [
            PredictionItem(SK_ID_CURR=int(rec["SK_ID_CURR"]), TARGET=float(rec["TARGET"]))
            for rec in predictions.to_dict(orient="records")
        ]
        LOGGER.info("Returning %d predictions", len(items))
        bundle = self.engine.bundle
        active = bundle.get("model", bundle)
        return PredictResponse(
            predictions=items,
            model_version=self.model_version,
            request_count=len(items),
            schema_version=str(bundle.get("schema_version", "1.0")),
            calibration_method=active.get("calibration_method") or bundle.get("calibration_method"),
            threshold=float(bundle["threshold"]) if bundle.get("threshold") is not None else None,
        )

    def _predict_batched(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= self.inference_batch_size:
            return self.engine.predict_dataframe(frame)
        parts = []
        for start in range(0, len(frame), self.inference_batch_size):
            chunk = frame.iloc[start : start + self.inference_batch_size]
            parts.append(self.engine.predict_dataframe(chunk))
        return pd.concat(parts, axis=0, ignore_index=True)
