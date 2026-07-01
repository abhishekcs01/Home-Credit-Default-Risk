from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InferenceRecord(BaseModel):
    SK_ID_CURR: int = Field(..., ge=1)
    features: dict[str, float | int | None] = Field(default_factory=dict, max_length=2048)
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def collect_dynamic_features(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            raise TypeError("Each record must be an object.")
        if "SK_ID_CURR" not in raw:
            raise ValueError("SK_ID_CURR is required for each record.")
        extras = {k: v for k, v in raw.items() if k != "SK_ID_CURR"}
        return {"SK_ID_CURR": raw["SK_ID_CURR"], "features": extras}

    @model_validator(mode="after")
    def validate_features(self) -> "InferenceRecord":
        cleaned: dict[str, float | int | None] = {}
        for key, value in self.features.items():
            if not key or not isinstance(key, str):
                raise ValueError("Feature keys must be non-empty strings.")
            if value is None:
                cleaned[key] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Feature '{key}' must be numeric or null.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Feature '{key}' must be finite.")
            if abs(float(value)) > 1e12:
                raise ValueError(f"Feature '{key}' is outside allowed numeric range.")
            cleaned[key] = value
        self.features = cleaned
        return self

    def to_feature_dict(self) -> dict[str, Any]:
        return {"SK_ID_CURR": self.SK_ID_CURR, **self.features}


class PredictRequest(BaseModel):
    records: list[InferenceRecord] = Field(..., min_length=1, max_length=2048)


class PredictionItem(BaseModel):
    SK_ID_CURR: int
    TARGET: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    model_version: str
    request_count: int
    schema_version: str = "1.0"
    calibration_method: str | None = None
    threshold: float | None = None


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    inference_ready: bool = False
    bundle_path_exists: bool = False
