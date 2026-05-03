import numpy as np
import pandas as pd


def engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    inc = out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    credit = out["AMT_CREDIT"].replace(0, np.nan)
    ann = out["AMT_ANNUITY"].replace(0, np.nan)

    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / inc
    members = out["CNT_FAM_MEMBERS"].replace(0, np.nan)
    out["CHILDREN_RATIO"] = out["CNT_CHILDREN"] / members

    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    out["EMPLOYMENT_YEARS"] = -out["DAYS_EMPLOYED"] / 365.25
    out["EMPLOYMENT_AGE_RATIO"] = out["EMPLOYMENT_YEARS"] / out["AGE_YEARS"].replace(0, np.nan)

    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / inc
    out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / credit
    out["CREDIT_TERM"] = out["AMT_CREDIT"] / ann

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    present = [c for c in ext_cols if c in out.columns]
    if present:
        out["EXT_SOURCE_MEAN"] = out[present].mean(axis=1)
        out["EXT_SOURCE_MIN"] = out[present].min(axis=1)
        out["EXT_SOURCE_MAX"] = out[present].max(axis=1)
        out["EXT_SOURCE_STD"] = out[present].std(axis=1)

    return out.replace([np.inf, -np.inf], np.nan)

