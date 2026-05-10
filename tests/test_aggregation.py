import numpy as np
import pandas as pd

from src.aggregation import _downcast_numeric_columns, aggregate_bureau, aggregate_bureau_balance, merge_stage


def test_bureau_aggregation_returns_sk_id_curr():
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2],
            "SK_ID_BUREAU": [11, 12, 21],
            "DAYS_CREDIT": [-100, -200, -50],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 500.0],
        }
    )
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [11, 11, 12, 21],
            "MONTHS_BALANCE": [-1, -2, -1, -1],
            "STATUS": ["0", "1", "0", "2"],
        }
    )
    bb_agg = aggregate_bureau_balance(bb)
    buro_agg = aggregate_bureau(bureau, bb_agg)
    assert "SK_ID_CURR" in buro_agg.columns
    assert buro_agg["SK_ID_CURR"].nunique() == 2
    assert "BURO_BB_WORST_DELINQ_STATUS_max" in buro_agg.columns
    assert "BURO_BB_MAX_DELINQ_STREAK_max" in buro_agg.columns


def test_merge_stage_app_only_identity():
    app = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]})
    out = merge_stage(app, "app")
    assert list(out.columns) == list(app.columns)
    assert out["SK_ID_CURR"].tolist() == app["SK_ID_CURR"].tolist()
    assert out["TARGET"].tolist() == app["TARGET"].tolist()


def test_bureau_balance_delinquency_features_exist():
    bb = pd.DataFrame(
        {
            "SK_ID_BUREAU": [11, 11, 11, 12, 12],
            "MONTHS_BALANCE": [-1, -2, -3, -1, -2],
            "STATUS": ["0", "3", "2", "0", "0"],
        }
    )
    out = aggregate_bureau_balance(bb)
    expected_cols = {
        "BB_WORST_DELINQ_STATUS",
        "BB_MAX_DELINQ_STREAK",
        "BB_SEVERE_DELINQ_FREQ",
        "BB_W3_SEVERE_DELINQ_SHARE",
    }
    assert expected_cols.issubset(set(out.columns))
    row_11 = out.loc[out["SK_ID_BUREAU"] == 11].iloc[0]
    assert row_11["BB_WORST_DELINQ_STATUS"] >= 3
    assert row_11["BB_MAX_DELINQ_STREAK"] >= 2


def test_downcast_numeric_columns_shrinks_64bit_types():
    frame = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2],
            "F64": np.array([1.0, 2.0], dtype=np.float64),
            "I64": np.array([10, 20], dtype=np.int64),
        }
    )
    out = _downcast_numeric_columns(frame.copy())
    assert str(out["F64"].dtype) == "float32"
    assert str(out["I64"].dtype) in {"int8", "int16", "int32"}

