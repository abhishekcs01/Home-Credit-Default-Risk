from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel


class FeatureColumnSpec(BaseModel):
    name: str
    dtype: str
    nullable: bool
    role: Literal["id", "target", "feature"]


class FeatureSchema(BaseModel):
    version: str
    columns: list[FeatureColumnSpec]

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame, *, version: str = "v1") -> "FeatureSchema":
        cols: list[FeatureColumnSpec] = []
        for col in frame.columns:
            role: Literal["id", "target", "feature"]
            if col == "SK_ID_CURR":
                role = "id"
            elif col == "TARGET":
                role = "target"
            else:
                role = "feature"
            cols.append(
                FeatureColumnSpec(
                    name=col,
                    dtype=str(frame[col].dtype),
                    nullable=bool(frame[col].isna().any()),
                    role=role,
                )
            )
        return cls(version=version, columns=cols)
