from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src import config

MISSING_INDICATOR_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "OCCUPATION_TYPE",
]


def replace_day_sentinel(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out.loc[out[c] == config.SENTINEL_DAYS, c] = np.nan
    return out


def _dmp_add_missing_indicators(frame: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in frame.columns:
            frame[f"{col}_was_missing"] = frame[col].isna().astype(np.int8)


def _dmp_fill_categorical_inplace(frame: pd.DataFrame, cat_cols: list[str]) -> None:
    for c in cat_cols:
        if frame[c].dtype.name == "category":
            if "Unknown" not in frame[c].cat.categories:
                frame[c] = frame[c].cat.add_categories(["Unknown"])
            frame[c] = frame[c].fillna("Unknown")
        else:
            frame[c] = frame[c].fillna("Unknown")


def _dmp_fill_categorical_features_inplace(features: pd.DataFrame, cat_cols: list[str]) -> None:
    for c in cat_cols:
        if features[c].dtype.name == "category":
            if "Unknown" not in features[c].cat.categories:
                try:
                    features[c] = features[c].cat.add_categories(["Unknown"])
                except (TypeError, ValueError):
                    features[c] = features[c].astype(str)
            features[c] = features[c].fillna("Unknown")
        else:
            features[c] = features[c].fillna("Unknown")


def _dmp_apply_label_encoders_inplace(
    x_enc: pd.DataFrame, high_card: list[str], encoders: dict[str, LabelEncoder]
) -> None:
    for c in high_card:
        le = encoders[c]
        classes = list(le.classes_)
        unk = len(classes)
        mapping = {lab: i for i, lab in enumerate(classes)}
        x_enc[c] = x_enc[c].astype(str).map(lambda x, m=mapping, u=unk: m.get(x, u)).astype(np.float32)


def _dmp_make_ohe() -> OneHotEncoder | None:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _dmp_hstack_numeric_ohe(
    x_enc: pd.DataFrame,
    num_cols: list[str],
    high_card: list[str],
    low_card: list[str],
    ohe: OneHotEncoder | None,
) -> np.ndarray:
    numeric_block = x_enc[num_cols + high_card].to_numpy(dtype=np.float32)
    if low_card and ohe is not None:
        ohe_matrix = ohe.transform(x_enc[low_card])
        return np.hstack([numeric_block, np.asarray(ohe_matrix, dtype=np.float32)])
    return numeric_block


class DesignMatrixPreprocessor:
    def __init__(
        self,
        ohe_max_categories: int = config.OHE_MAX_CATEGORIES,
        miss_drop_threshold: float = config.MISS_DROP_THRESHOLD,
    ):
        self.ohe_max_categories = ohe_max_categories
        self.miss_drop_threshold = miss_drop_threshold

    def fit(self, df: pd.DataFrame) -> "DesignMatrixPreprocessor":
        frame = df.replace([np.inf, -np.inf], np.nan)
        miss_frac = frame.isna().mean()
        self.drop_cols_ = [
            c for c in miss_frac[miss_frac > self.miss_drop_threshold].index if c not in ("TARGET", "SK_ID_CURR")
        ]
        frame = frame.drop(columns=self.drop_cols_, errors="ignore")
        _dmp_add_missing_indicators(frame, MISSING_INDICATOR_COLS)

        if "TARGET" not in frame.columns:
            raise ValueError("Training frame must contain TARGET.")

        self.y_ = frame["TARGET"].astype(int).to_numpy()
        features = frame.drop(columns=["TARGET"])
        if "SK_ID_CURR" in features.columns:
            features = features.drop(columns=["SK_ID_CURR"])

        self.raw_columns_ = features.columns.tolist()
        self.cat_cols_ = features.select_dtypes(include=["category", "object"]).columns.tolist()
        self.num_cols_ = features.select_dtypes(include=[np.number]).columns.tolist()

        self.medians_ = {}
        for c in self.num_cols_:
            med = features[c].median()
            self.medians_[c] = 0.0 if pd.isna(med) else float(med)

        filled = features.copy()
        for c in self.num_cols_:
            filled[c] = filled[c].fillna(self.medians_[c]).replace([np.inf, -np.inf], np.nan)
        for c in self.num_cols_:
            filled[c] = filled[c].fillna(self.medians_[c])
        _dmp_fill_categorical_inplace(filled, self.cat_cols_)

        self.low_card_, self.high_card_ = [], []
        for c in self.cat_cols_:
            nuniq = filled[c].nunique(dropna=False)
            (self.low_card_ if nuniq <= self.ohe_max_categories else self.high_card_).append(c)

        self.label_encoders_ = {}
        x_enc = filled.copy()
        for c in self.high_card_:
            le = LabelEncoder()
            x_enc[c] = le.fit_transform(x_enc[c].astype(str))
            self.label_encoders_[c] = le

        self.ohe_ = _dmp_make_ohe()
        if self.low_card_:
            self.ohe_.fit(x_enc[self.low_card_])
        else:
            self.ohe_ = None

        ohe_names = list(self.ohe_.get_feature_names_out(self.low_card_)) if self.low_card_ and self.ohe_ else []
        self.feature_names_ = self.num_cols_ + self.high_card_ + ohe_names
        self.X_train_ = _dmp_hstack_numeric_ohe(x_enc, self.num_cols_, self.high_card_, self.low_card_, self.ohe_)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Call fit() before transform().")

        frame = df.replace([np.inf, -np.inf], np.nan)
        frame = frame.drop(columns=self.drop_cols_, errors="ignore")
        _dmp_add_missing_indicators(frame, MISSING_INDICATOR_COLS)
        features = frame.drop(columns=["TARGET"], errors="ignore")
        if "SK_ID_CURR" in features.columns:
            features = features.drop(columns=["SK_ID_CURR"])

        for col in self.raw_columns_:
            if col not in features.columns:
                features[col] = np.nan
        drop_extra = [c for c in features.columns if c not in self.raw_columns_]
        if drop_extra:
            features = features.drop(columns=drop_extra)
        features = features[self.raw_columns_]

        for c in self.num_cols_:
            features[c] = features[c].fillna(self.medians_[c]).replace([np.inf, -np.inf], np.nan)
        for c in self.num_cols_:
            features[c] = features[c].fillna(self.medians_[c])
        _dmp_fill_categorical_features_inplace(features, self.cat_cols_)

        x_enc = features.copy()
        _dmp_apply_label_encoders_inplace(x_enc, self.high_card_, self.label_encoders_)
        return _dmp_hstack_numeric_ohe(x_enc, self.num_cols_, self.high_card_, self.low_card_, self.ohe_)


def dataframe_to_design_matrix(
    df: pd.DataFrame,
    *,
    ohe_max_categories: int = config.OHE_MAX_CATEGORIES,
    miss_drop_threshold: float = config.MISS_DROP_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prep = DesignMatrixPreprocessor(
        ohe_max_categories=ohe_max_categories,
        miss_drop_threshold=miss_drop_threshold,
    )
    prep.fit(df)
    return prep.X_train_, prep.y_, prep.feature_names_

