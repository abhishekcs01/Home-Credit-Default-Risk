from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src import config

MISSING_INDICATOR_COLS = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "OCCUPATION_TYPE",
]
TARGET_ENCODING_PRIORITY_COLS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_INCOME_TYPE",
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


def _fill_categorical_inplace(frame: pd.DataFrame, cat_cols: list[str]) -> None:
    for c in cat_cols:
        if frame[c].dtype.name == "category":
            if "Unknown" not in frame[c].cat.categories:
                try:
                    frame[c] = frame[c].cat.add_categories(["Unknown"])
                except (TypeError, ValueError):
                    frame[c] = frame[c].astype(str)
            frame[c] = frame[c].fillna("Unknown")
        else:
            frame[c] = frame[c].fillna("Unknown")


def _make_ohe() -> OneHotEncoder | None:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def _build_target_encoding(
    series: pd.Series,
    y: np.ndarray,
    *,
    min_samples_leaf: int,
    smoothing: float,
) -> tuple[dict[str, float], float]:
    s = series.astype(str)
    target = pd.Series(y, index=series.index, dtype=float)
    tmp = pd.DataFrame({"cat": s, "target": target}).groupby("cat")["target"].agg(["mean", "count"])
    prior = float(target.mean())
    smooth = 1.0 / (1.0 + np.exp(-(tmp["count"] - min_samples_leaf) / smoothing))
    encoded = prior * (1.0 - smooth) + tmp["mean"] * smooth
    return encoded.astype(float).to_dict(), prior


class DesignMatrixPreprocessor:
    def __init__(
        self,
        ohe_max_categories: int = config.OHE_MAX_CATEGORIES,
        miss_drop_threshold: float = config.MISS_DROP_THRESHOLD,
        te_min_samples_leaf: int = 80,
        te_smoothing: float = 25.0,
    ):
        self.ohe_max_categories = ohe_max_categories
        self.miss_drop_threshold = miss_drop_threshold
        self.te_min_samples_leaf = te_min_samples_leaf
        self.te_smoothing = te_smoothing

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.replace([np.inf, -np.inf], np.nan)
        frame = frame.drop(columns=self.drop_cols_, errors="ignore")
        _dmp_add_missing_indicators(frame, MISSING_INDICATOR_COLS)
        features = frame.drop(columns=["TARGET"], errors="ignore")
        if "SK_ID_CURR" in features.columns:
            features = features.drop(columns=["SK_ID_CURR"])
        return features

    def _align_feature_columns(self, features: pd.DataFrame) -> pd.DataFrame:
        for col in self.raw_columns_:
            if col not in features.columns:
                features[col] = np.nan
        drop_extra = [c for c in features.columns if c not in self.raw_columns_]
        if drop_extra:
            features = features.drop(columns=drop_extra)
        return features[self.raw_columns_]

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
        y = frame["TARGET"].astype(int).to_numpy()
        self.y_ = y

        features = frame.drop(columns=["TARGET"])
        if "SK_ID_CURR" in features.columns:
            features = features.drop(columns=["SK_ID_CURR"])

        self.raw_columns_ = features.columns.tolist()
        self.cat_cols_ = features.select_dtypes(include=["category", "object", "string"]).columns.tolist()
        self.num_cols_ = features.select_dtypes(include=[np.number]).columns.tolist()

        self.medians_ = {}
        for c in self.num_cols_:
            med = features[c].median()
            self.medians_[c] = 0.0 if pd.isna(med) else float(med)

        filled = features.copy()
        for c in self.num_cols_:
            filled[c] = filled[c].fillna(self.medians_[c]).replace([np.inf, -np.inf], np.nan)
            filled[c] = filled[c].fillna(self.medians_[c])
        _fill_categorical_inplace(filled, self.cat_cols_)

        self.low_card_, self.high_card_ = [], []
        for c in self.cat_cols_:
            nuniq = filled[c].nunique(dropna=False)
            if c in TARGET_ENCODING_PRIORITY_COLS or nuniq > self.ohe_max_categories:
                self.high_card_.append(c)
            else:
                self.low_card_.append(c)

        self.target_encoding_cols_ = list(self.high_card_)
        self.target_prior_ = float(np.mean(y))
        self.target_encoding_maps_ = {}
        for c in self.target_encoding_cols_:
            mp, _ = _build_target_encoding(
                filled[c],
                y,
                min_samples_leaf=self.te_min_samples_leaf,
                smoothing=self.te_smoothing,
            )
            self.target_encoding_maps_[c] = mp

        self.ohe_ = _make_ohe()
        if self.low_card_:
            self.ohe_.fit(filled[self.low_card_])
            ohe_names = list(self.ohe_.get_feature_names_out(self.low_card_))
        else:
            self.ohe_ = None
            ohe_names = []

        self.te_feature_names_ = [f"{c}__TE" for c in self.target_encoding_cols_]
        self.feature_names_ = self.num_cols_ + self.te_feature_names_ + ohe_names
        self.X_train_ = self.transform(df)
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "raw_columns_", None):
            raise RuntimeError("Call fit() before transform().")
        features = self._prepare_features(df.copy())
        features = self._align_feature_columns(features)

        for c in self.num_cols_:
            features[c] = features[c].fillna(self.medians_[c]).replace([np.inf, -np.inf], np.nan)
            features[c] = features[c].fillna(self.medians_[c])
        _fill_categorical_inplace(features, self.cat_cols_)

        blocks: list[np.ndarray] = [features[self.num_cols_].to_numpy(dtype=np.float32)] if self.num_cols_ else []
        if self.target_encoding_cols_:
            te_block = np.zeros((len(features), len(self.target_encoding_cols_)), dtype=np.float32)
            for i, c in enumerate(self.target_encoding_cols_):
                mp = self.target_encoding_maps_[c]
                te_block[:, i] = (
                    features[c].astype(str).map(mp).fillna(self.target_prior_).astype(np.float32).to_numpy(copy=False)
                )
            blocks.append(te_block)
        if self.low_card_ and self.ohe_ is not None:
            ohe_block = self.ohe_.transform(features[self.low_card_])
            blocks.append(np.asarray(ohe_block, dtype=np.float32))

        if not blocks:
            return np.empty((len(features), 0), dtype=np.float32)
        return blocks[0] if len(blocks) == 1 else np.hstack(blocks).astype(np.float32, copy=False)


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

