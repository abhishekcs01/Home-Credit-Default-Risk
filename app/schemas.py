from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InferenceRecord(BaseModel):
    SK_ID_CURR: int = Field(..., ge=1, description="Unique Home Credit application id.")
    model_config = ConfigDict(extra="allow")

    def to_feature_dict(self) -> dict[str, Any]:
        return self.model_dump()


class PredictRequest(BaseModel):
    records: list[InferenceRecord] = Field(..., min_length=1, max_length=2048)


class PredictionItem(BaseModel):
    SK_ID_CURR: int
    TARGET: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    model_version: str
    request_count: int


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
