import numpy as np
import pandas as pd


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_safe = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(num, errors="coerce") / den_safe


def trend_delta(recent: pd.Series, baseline: pd.Series) -> pd.Series:
    return pd.to_numeric(recent, errors="coerce") - pd.to_numeric(baseline, errors="coerce")


def coefficient_of_variation(mean: pd.Series, std: pd.Series) -> pd.Series:
    return safe_ratio(pd.to_numeric(std, errors="coerce"), pd.to_numeric(mean, errors="coerce").abs() + 1e-6)


def recency_lift(recency_metric: pd.Series, baseline_metric: pd.Series) -> pd.Series:
    return safe_ratio(pd.to_numeric(recency_metric, errors="coerce"), pd.to_numeric(baseline_metric, errors="coerce"))


def as_float32(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)


def _series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _add_ratio_feature(out: pd.DataFrame, name: str, num_col: str, den_col: str) -> str:
    out[name] = safe_ratio(_series_or_nan(out, num_col), _series_or_nan(out, den_col))
    return name


def _add_trend_feature(out: pd.DataFrame, name: str, recent_col: str, baseline_col: str) -> str:
    out[name] = trend_delta(_series_or_nan(out, recent_col), _series_or_nan(out, baseline_col))
    return name


def _add_cv_feature(out: pd.DataFrame, name: str, mean_col: str, std_col: str) -> str:
    out[name] = coefficient_of_variation(_series_or_nan(out, mean_col), _series_or_nan(out, std_col))
    return name


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_features: list[str] = []
    inc = out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit = out["AMT_CREDIT"].replace(0, np.nan)
    ann = out["AMT_ANNUITY"].replace(0, np.nan)

    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / inc
    new_features.append("ANNUITY_INCOME_RATIO")
    members = out["CNT_FAM_MEMBERS"].replace(0, np.nan)
    out["CHILDREN_RATIO"] = out["CNT_CHILDREN"] / members
    new_features.append("CHILDREN_RATIO")

    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["EMPLOYMENT_YEARS"] = -out["DAYS_EMPLOYED"] / 365.25
    out["EMPLOYMENT_AGE_RATIO"] = out["EMPLOYMENT_YEARS"] / out["AGE_YEARS"].replace(0, np.nan)
    new_features.extend(["AGE_YEARS", "EMPLOYMENT_YEARS", "EMPLOYMENT_AGE_RATIO"])

    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / inc
    out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / credit
    out["CREDIT_TERM"] = out["AMT_CREDIT"] / ann
    new_features.extend(["CREDIT_INCOME_RATIO", "ANNUITY_CREDIT_RATIO", "CREDIT_TERM"])

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    present = [c for c in ext_cols if c in out.columns]
    if present:
        out["EXT_SOURCE_MEAN"] = out[present].mean(axis=1)
        out["EXT_SOURCE_MIN"] = out[present].min(axis=1)
        out["EXT_SOURCE_MAX"] = out[present].max(axis=1)
        out["EXT_SOURCE_STD"] = out[present].std(axis=1)
        new_features.extend(["EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_MAX", "EXT_SOURCE_STD"])

    new_features.append(
        _add_trend_feature(
            out, "INST_LATE_TREND_90_365", "INST_W90_INST_PAYMENT_DIFF_POS_mean", "INST_W365_INST_PAYMENT_DIFF_POS_mean"
        )
    )
    new_features.append(_add_trend_feature(out, "POS_DPD_TREND_3_12", "POS_W3_SK_DPD_mean", "POS_W12_SK_DPD_mean"))
    new_features.append(
        _add_trend_feature(out, "CC_UTIL_TREND_3_12", "CC_W3_CC_UTILIZATION_mean", "CC_W12_CC_UTILIZATION_mean")
    )
    new_features.append(
        _add_trend_feature(
            out, "BURO_STATUS_TREND_3_12", "BURO_BB_W3_STATUS_NUM_mean_mean", "BURO_BB_W12_STATUS_NUM_mean_mean"
        )
    )

    out["INST_RECENCY_LATE_LIFT"] = recency_lift(
        _series_or_nan(out, "INST_RECENCY_WMEAN_LATE_DAYS"),
        _series_or_nan(out, "INST_INST_PAYMENT_DIFF_POS_mean"),
    )
    out["POS_RECENCY_DPD_LIFT"] = recency_lift(
        _series_or_nan(out, "POS_RECENCY_WMEAN_DPD"),
        _series_or_nan(out, "POS_SK_DPD_mean"),
    )
    out["CC_RECENCY_UTIL_LIFT"] = recency_lift(
        _series_or_nan(out, "CC_RECENCY_WMEAN_UTIL"),
        _series_or_nan(out, "CC_CC_UTILIZATION_mean"),
    )
    out["BURO_RECENCY_STATUS_LIFT"] = recency_lift(
        _series_or_nan(out, "BURO_BB_RECENCY_WMEAN_STATUS_mean"),
        _series_or_nan(out, "BURO_BB_STATUS_NUM_mean_mean"),
    )
    new_features.extend(
        [
            "INST_RECENCY_LATE_LIFT",
            "POS_RECENCY_DPD_LIFT",
            "CC_RECENCY_UTIL_LIFT",
            "BURO_RECENCY_STATUS_LIFT",
        ]
    )

    new_features.append(
        _add_ratio_feature(out, "SEVERE_DELINQ_TO_ANY_DELINQ_RATIO", "INST_SEVERE_LATE_PAYMENT_RATE", "INST_LATE_PAYMENT_RATE")
    )
    new_features.append(_add_ratio_feature(out, "POS_DELINQ_TO_CC_DELINQ_RATIO", "POS_DELINQ_FREQ", "CC_DELINQ_FREQ"))
    new_features.append(
        _add_ratio_feature(
            out,
            "BURO_SEVERE_TO_TOTAL_DELINQ_RATIO",
            "BURO_BB_SEVERE_DELINQ_FREQ_mean",
            "BURO_BB_BB_IS_DELINQ_mean_mean",
        )
    )
    new_features.append(
        _add_ratio_feature(out, "INST_PAYMENT_COVERAGE_RATIO", "INST_AMT_PAYMENT_sum", "INST_AMT_INSTALMENT_sum")
    )

    new_features.append(
        _add_cv_feature(out, "INST_PAYMENT_RATIO_CV", "INST_INST_PAYMENT_RATIO_mean", "INST_INST_PAYMENT_RATIO_std")
    )
    new_features.append(_add_cv_feature(out, "CC_UTILIZATION_CV", "CC_CC_UTILIZATION_mean", "CC_CC_UTILIZATION_std"))
    new_features.append(_add_cv_feature(out, "BURO_CREDIT_SUM_CV", "BURO_AMT_CREDIT_SUM_mean", "BURO_AMT_CREDIT_SUM_std"))
    new_features.append(_add_cv_feature(out, "PREV_APP_CREDIT_CV", "PREV_AMT_CREDIT_mean", "PREV_AMT_CREDIT_std"))

    new_features.append(_add_ratio_feature(out, "CREDIT_TO_PREV_CREDIT_RATIO", "AMT_CREDIT", "PREV_AMT_CREDIT_mean"))
    new_features.append(_add_ratio_feature(out, "ANNUITY_TO_PREV_ANNUITY_RATIO", "AMT_ANNUITY", "PREV_AMT_ANNUITY_mean"))
    new_features.append(_add_ratio_feature(out, "INCOME_TO_POS_DPD_RATIO", "AMT_INCOME_TOTAL", "POS_SK_DPD_mean"))
    new_features.append(_add_ratio_feature(out, "CREDIT_TO_CC_LIMIT_RATIO", "AMT_CREDIT", "CC_AMT_CREDIT_LIMIT_ACTUAL_mean"))
    new_features.append(_add_ratio_feature(out, "CREDIT_TO_BURO_DEBT_RATIO", "AMT_CREDIT", "BURO_AMT_CREDIT_SUM_DEBT_mean"))
    new_features.append(_add_ratio_feature(out, "INCOME_TO_INST_PAYMENT_RATIO", "AMT_INCOME_TOTAL", "INST_AMT_PAYMENT_mean"))

    new_features.append(
        _add_ratio_feature(out, "DEBT_ACCELERATION_RATIO", "BURO_AMT_CREDIT_SUM_DEBT_sum", "BURO_AMT_CREDIT_SUM_sum")
    )
    new_features.append(
        _add_trend_feature(out, "BURO_DEBT_TREND_RECENT_LONG", "BURO_BB_W3_STATUS_NUM_mean_mean", "BURO_BB_STATUS_NUM_mean_mean")
    )
    new_features.append(_add_ratio_feature(out, "PREV_REFUSAL_PRESSURE", "PREV_REFUSED_RATE", "PREV_APPROVED_RATE"))
    new_features.append(_add_ratio_feature(out, "CC_HIGH_UTIL_TO_POS_DELINQ", "CC_HIGH_UTILIZATION_FREQ", "POS_DELINQ_FREQ"))

    out["EXT_SPREAD"] = _series_or_nan(out, "EXT_SOURCE_MAX") - _series_or_nan(out, "EXT_SOURCE_MIN")
    out["EXT_MEAN_STD_RATIO"] = safe_ratio(_series_or_nan(out, "EXT_SOURCE_MEAN"), _series_or_nan(out, "EXT_SOURCE_STD"))
    out["INCOME_PER_FAMILY_MEMBER"] = safe_ratio(_series_or_nan(out, "AMT_INCOME_TOTAL"), members)
    out["CREDIT_PER_FAMILY_MEMBER"] = safe_ratio(_series_or_nan(out, "AMT_CREDIT"), members)
    new_features.extend(["EXT_SPREAD", "EXT_MEAN_STD_RATIO", "INCOME_PER_FAMILY_MEMBER", "CREDIT_PER_FAMILY_MEMBER"])

    # Interaction features (capacity × behavior)
    out["INCOME_X_CREDIT"] = _series_or_nan(out, "AMT_INCOME_TOTAL") * _series_or_nan(out, "AMT_CREDIT")
    out["INCOME_X_ANNUITY"] = _series_or_nan(out, "AMT_INCOME_TOTAL") * _series_or_nan(out, "AMT_ANNUITY")
    out["EMPLOYMENT_X_AGE"] = _series_or_nan(out, "EMPLOYMENT_YEARS") * _series_or_nan(out, "AGE_YEARS")
    out["CREDIT_X_FAMILY_SIZE"] = _series_or_nan(out, "AMT_CREDIT") * members
    buro_activity = _series_or_nan(out, "BURO_AMT_CREDIT_SUM_count").fillna(0) + _series_or_nan(
        out, "PREV_AMT_CREDIT_count"
    ).fillna(0)
    out["CREDIT_X_BUREAU_ACTIVITY"] = _series_or_nan(out, "AMT_CREDIT") * buro_activity
    new_features.extend(
        ["INCOME_X_CREDIT", "INCOME_X_ANNUITY", "EMPLOYMENT_X_AGE", "CREDIT_X_FAMILY_SIZE", "CREDIT_X_BUREAU_ACTIVITY"]
    )

    # Stability / growth indicators
    new_features.append(
        _add_trend_feature(out, "INST_PAYMENT_ACCEL", "INST_W90_INST_IS_LATE_mean", "INST_W365_INST_IS_LATE_mean")
    )
    out["CC_BALANCE_VOLATILITY"] = coefficient_of_variation(
        _series_or_nan(out, "CC_AMT_BALANCE_mean"), _series_or_nan(out, "CC_AMT_BALANCE_std")
    )
    out["BURO_CREDIT_VELOCITY"] = safe_ratio(
        _series_or_nan(out, "BURO_AMT_CREDIT_SUM_sum"),
        _series_or_nan(out, "BURO_DAYS_CREDIT_min").abs() + 1.0,
    )
    out["PREV_APP_FREQUENCY"] = _series_or_nan(out, "PREV_AMT_CREDIT_count")
    out["PREV_APP_RECENCY_DAYS"] = _series_or_nan(out, "PREV_DAYS_DECISION_min")
    new_features.extend(
        ["INST_PAYMENT_ACCEL", "CC_BALANCE_VOLATILITY", "BURO_CREDIT_VELOCITY", "PREV_APP_FREQUENCY", "PREV_APP_RECENCY_DAYS"]
    )

    out["INSTALLMENT_PAYMENT_STABILITY"] = safe_ratio(
        1.0 / (1.0 + _series_or_nan(out, "INST_INST_PAYMENT_RATIO_std").abs()),
        pd.Series(1.0, index=out.index),
    )
    out["INSTALLMENT_LATE_SEVERITY_INDEX"] = _series_or_nan(out, "INST_LATE_PAYMENT_RATE") * _series_or_nan(
        out, "INST_INST_PAYMENT_DIFF_POS_mean"
    )
    out["DELINQ_COMPOSITE_INDEX"] = (
        _series_or_nan(out, "POS_DELINQ_FREQ").fillna(0)
        + _series_or_nan(out, "CC_DELINQ_FREQ").fillna(0)
        + _series_or_nan(out, "INST_LATE_PAYMENT_RATE").fillna(0)
    )
    new_features.extend(
        [
            "INSTALLMENT_PAYMENT_STABILITY",
            "INSTALLMENT_LATE_SEVERITY_INDEX",
            "DELINQ_COMPOSITE_INDEX",
        ]
    )

    as_float32(out, [c for c in new_features if c in out.columns])
    out.attrs["feature_metadata"] = {
        "engineered_features": [c for c in new_features if c in out.columns],
        "n_engineered_features": int(len([c for c in new_features if c in out.columns])),
        "version": "phase3_v2",
    }

    return out.replace([np.inf, -np.inf], np.nan)
