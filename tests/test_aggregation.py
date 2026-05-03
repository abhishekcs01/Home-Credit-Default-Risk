import pandas as pd

from src.aggregation import aggregate_bureau, aggregate_bureau_balance, merge_stage


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


def test_merge_stage_app_only_identity():
    app = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]})
    out = merge_stage(app, "app")
    assert out.equals(app)

