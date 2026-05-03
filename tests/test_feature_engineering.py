import numpy as np
import pandas as pd

from src.feature_engineering import engineer_application_features


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
    }
    assert expected.issubset(set(out.columns))
    assert np.isfinite(out["CREDIT_INCOME_RATIO"]).all()

