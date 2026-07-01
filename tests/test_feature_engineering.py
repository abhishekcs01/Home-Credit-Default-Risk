import numpy as np
import pandas as pd

from src.features.build_features import engineer_application_features


def test_engineer_application_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [100000.0, 200000.0],
            "AMT_CREDIT": [500000.0, 300000.0],
            "AMT_ANNUITY": [25000.0, 15000.0],
            "CNT_CHILDREN": [1, 0],
            "CNT_FAM_MEMBERS": [3, 2],
            "DAYS_BIRTH": [-12000, -14000],
            "DAYS_EMPLOYED": [-2000, -1000],
            "EXT_SOURCE_1": [0.2, 0.4],
            "EXT_SOURCE_2": [0.3, 0.5],
            "EXT_SOURCE_3": [0.1, 0.9],
            "POS_DELINQ_FREQ": [0.1, 0.3],
            "CC_DELINQ_FREQ": [0.2, 0.1],
            "INST_LATE_PAYMENT_RATE": [0.4, 0.2],
        }
    )
    out = engineer_application_features(df)
    expected = {
        "ANNUITY_INCOME_RATIO",
        "CHILDREN_RATIO",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "EMPLOYMENT_AGE_RATIO",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_CREDIT_RATIO",
        "CREDIT_TERM",
        "EXT_SOURCE_MEAN",
        "DELINQ_COMPOSITE_INDEX",
        "EXT_SPREAD",
        "INCOME_PER_FAMILY_MEMBER",
    }
    assert expected.issubset(set(out.columns))
    assert np.isfinite(out["CREDIT_INCOME_RATIO"]).all()
    assert out.attrs["feature_metadata"]["n_engineered_features"] >= len(expected)


def test_engineer_application_features_zero_income_yields_nan_ratios() -> None:
    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [0.0],
            "AMT_CREDIT": [100000.0],
            "AMT_ANNUITY": [10000.0],
            "CNT_CHILDREN": [0],
            "CNT_FAM_MEMBERS": [1],
            "DAYS_BIRTH": [-10000],
            "DAYS_EMPLOYED": [-500],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.5],
            "EXT_SOURCE_3": [0.5],
            "POS_DELINQ_FREQ": [0.0],
            "CC_DELINQ_FREQ": [0.0],
            "INST_LATE_PAYMENT_RATE": [0.0],
        }
    )
    out = engineer_application_features(df)
    assert pd.isna(out["CREDIT_INCOME_RATIO"].iloc[0])
    assert pd.isna(out["ANNUITY_INCOME_RATIO"].iloc[0])


def test_engineer_application_features_creates_cross_table_ratios_when_columns_present() -> None:
    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [120000.0],
            "AMT_CREDIT": [360000.0],
            "AMT_ANNUITY": [18000.0],
            "CNT_CHILDREN": [1],
            "CNT_FAM_MEMBERS": [3],
            "DAYS_BIRTH": [-13000],
            "DAYS_EMPLOYED": [-3000],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.6],
            "EXT_SOURCE_3": [0.4],
            "PREV_AMT_CREDIT_mean": [300000.0],
            "CC_AMT_CREDIT_LIMIT_ACTUAL_mean": [200000.0],
            "BURO_AMT_CREDIT_SUM_DEBT_mean": [100000.0],
            "INST_AMT_PAYMENT_mean": [15000.0],
            "POS_DELINQ_FREQ": [0.2],
            "CC_DELINQ_FREQ": [0.1],
            "INST_LATE_PAYMENT_RATE": [0.3],
        }
    )
    out = engineer_application_features(df)
    assert "CREDIT_TO_PREV_CREDIT_RATIO" in out.columns
    assert "CREDIT_TO_CC_LIMIT_RATIO" in out.columns
    assert "CREDIT_TO_BURO_DEBT_RATIO" in out.columns
    assert out["CREDIT_TO_PREV_CREDIT_RATIO"].iloc[0] > 1.0
