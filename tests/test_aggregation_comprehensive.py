from __future__ import annotations

import pandas as pd

from src.data.aggregation import (
    _balance_table_window_agg,
    _downcast_numeric_columns,
    _installments_window_agg,
    _merge_left_on_curr,
    _months_recent_mask,
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_credit_card,
    aggregate_installments,
    aggregate_pos_cash,
    aggregate_previous_application,
    merge_stage,
)


def _app() -> pd.DataFrame:
    return pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "AMT_INCOME_TOTAL": [100000.0, 120000.0]})


def _bureau_balance() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 10, 11, 11, 12],
            "MONTHS_BALANCE": [-1, -3, -2, -8, -1],
            "STATUS": ["0", "3", "1", "2", "C"],
        }
    )


def _bureau() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [10, 11, 12],
            "SK_ID_CURR": [1, 1, 2],
            "DAYS_CREDIT": [-10, -365243, -20],
            "AMT_CREDIT_SUM": [1000.0, 2000.0, 1500.0],
            "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
            "CREDIT_TYPE": ["Consumer credit", "Credit card", "Mortgage"],
        }
    )


def _previous() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_PREV": [100, 101, 102],
            "SK_ID_CURR": [1, 1, 2],
            "DAYS_DECISION": [-5, -40, -70],
            "AMT_APPLICATION": [5000.0, 8000.0, 7000.0],
            "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Canceled"],
            "FLAG_LAST_APPL_PER_CONTRACT": ["Y", "N", "Y"],
        }
    )


def _installments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2],
            "SK_ID_PREV": [100, 101, 102, 103],
            "DAYS_ENTRY_PAYMENT": [-2, -200, -10, -500],
            "DAYS_INSTALMENT": [-5, -180, -20, -530],
            "AMT_PAYMENT": [100.0, 80.0, 60.0, 90.0],
            "AMT_INSTALMENT": [100.0, 100.0, 80.0, 100.0],
        }
    )


def _pos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2],
            "SK_ID_PREV": [100, 101, 102, 103],
            "MONTHS_BALANCE": [-1, -4, -2, -7],
            "SK_DPD": [0, 5, 0, 2],
            "CNT_INSTALMENT_FUTURE": [2, 1, 3, 2],
        }
    )


def _cc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 1, 2, 2],
            "SK_ID_PREV": [100, 101, 102, 103],
            "MONTHS_BALANCE": [-1, -4, -2, -7],
            "AMT_BALANCE": [90.0, 50.0, 20.0, 80.0],
            "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0, 100.0, 100.0],
            "SK_DPD": [0, 2, 0, 10],
        }
    )


def test_core_helpers_cover_empty_and_non_empty_paths():
    empty = pd.DataFrame()
    assert _downcast_numeric_columns(empty).empty
    assert _months_recent_mask(pd.Series([-1, -12, -15]), 12).tolist() == [True, True, False]

    left = pd.DataFrame({"SK_ID_CURR": [1, 2], "X": [1.0, 2.0]})
    right = pd.DataFrame({"SK_ID_CURR": [1, 1, 2], "Y": [3.0, 9.0, 4.0]})
    merged = _merge_left_on_curr(left, right)
    assert "Y" in merged.columns
    assert float(merged.loc[merged["SK_ID_CURR"] == 1, "Y"].iloc[0]) in {3.0, 9.0}
    assert _merge_left_on_curr(left, pd.DataFrame()).equals(left)


def test_individual_aggregators_create_expected_features():
    bb = aggregate_bureau_balance(_bureau_balance())
    assert "BB_RECENCY_WMEAN_STATUS" in bb.columns
    assert "BB_MAX_DELINQ_STREAK" in bb.columns
    assert not bb.empty

    bu = aggregate_bureau(_bureau(), bb)
    assert "BURO_AMT_CREDIT_SUM_mean" in bu.columns
    assert any(c.startswith("BURO_CNT_ACTIVE_") for c in bu.columns)
    assert "BURO_N_CREDIT_TYPES" in bu.columns

    prev = aggregate_previous_application(_previous())
    assert "PREV_APPROVED_RATE" in prev.columns
    assert "PREV_HAS_REFUSAL" in prev.columns
    assert "PREV_LAST_APPL_PER_CONTRACT_RATE" in prev.columns

    inst = aggregate_installments(_installments())
    assert "INST_LATE_PAYMENT_RATE" in inst.columns
    assert "INST_RECENCY_WMEAN_LATE_DAYS" in inst.columns
    assert any(c.startswith("INST_W90_") for c in inst.columns)

    pos = aggregate_pos_cash(_pos())
    assert "POS_DELINQ_FREQ" in pos.columns
    assert "POS_MAX_SK_DPD" in pos.columns
    assert any(c.startswith("POS_W3_") for c in pos.columns)

    cc = aggregate_credit_card(_cc())
    assert "CC_HIGH_UTILIZATION_FREQ" in cc.columns
    assert "CC_DELINQ_FREQ" in cc.columns
    assert "CC_RECENCY_WMEAN_UTIL" in cc.columns


def test_window_aggregators_and_merge_stage_full_pipeline():
    inst = _installments()
    w = _installments_window_agg(inst, 365, "IW_")
    assert "SK_ID_CURR" in w.columns

    pos = _pos()
    bw = _balance_table_window_agg(pos, "POS_", 6, "SK_ID_CURR")
    assert "SK_ID_CURR" in bw.columns

    full = merge_stage(
        _app(),
        "full",
        bureau=_bureau(),
        bureau_balance=_bureau_balance(),
        previous_application=_previous(),
        installments_payments=_installments(),
        pos_cash_balance=_pos(),
        credit_card_balance=_cc(),
    )
    assert set([1, 2]) == set(full["SK_ID_CURR"].tolist())
    assert any(c.startswith("BURO_") for c in full.columns)
    assert any(c.startswith("PREV_") for c in full.columns)
    assert any(c.startswith("INST_") for c in full.columns)
    assert any(c.startswith("POS_") for c in full.columns)
    assert any(c.startswith("CC_") for c in full.columns)
