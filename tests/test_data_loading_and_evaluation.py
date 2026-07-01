from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_data as data_loading
from src.models.evaluate import (
    _prepare_importance_table,
    feature_bucket,
    plot_ensemble_feature_importance,
    plot_feature_importance,
    plot_pr_curve,
    plot_roc_curve,
)


class _DummyModel:
    def feature_importance(self, importance_type: str = "gain"):
        assert importance_type == "gain"
        return np.array([3.0, 1.0, 2.0], dtype=float)


def test_data_loading_paths_and_memory_toggle(monkeypatch):
    calls: list[str] = []

    def fake_read_csv(path):
        calls.append(str(path))
        return pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    def fake_reduce(df):
        out = df.copy()
        out["REDUCED"] = 1
        return out

    monkeypatch.setattr(data_loading, "raw_data_available", lambda: True)
    monkeypatch.setattr(data_loading.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(data_loading, "reduce_mem_usage", fake_reduce)

    tr = data_loading.load_application_train(optimize_memory=True)
    te = data_loading.load_application_test(optimize_memory=False)
    aux = data_loading.load_auxiliary_tables(optimize_memory=True)
    aux_no_opt = data_loading.load_auxiliary_tables(optimize_memory=False)

    assert "REDUCED" in tr.columns
    assert "REDUCED" not in te.columns
    assert set(aux) == {
        "bureau",
        "bureau_balance",
        "previous_application",
        "installments_payments",
        "pos_cash_balance",
        "credit_card_balance",
    }
    assert all("REDUCED" in df.columns for df in aux.values())
    assert all("REDUCED" not in df.columns for df in aux_no_opt.values())
    assert len(calls) == 14  # 1 train + 1 test + 6 + 6 aux loads


def test_evaluation_plots_and_importance_helpers(tmp_path: Path):
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    y_score = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=float)

    roc_path = tmp_path / "figs" / "roc.png"
    pr_path = tmp_path / "figs" / "pr.png"
    fi_path = tmp_path / "figs" / "fi.png"
    efi_path = tmp_path / "figs" / "efi.png"

    roc_fig = plot_roc_curve(y_true, y_score, output_path=roc_path, model_name="unit")
    pr_fig = plot_pr_curve(y_true, y_score, output_path=pr_path, model_name="unit")
    fi_fig, fi_top = plot_feature_importance(
        _DummyModel(),
        feature_names=["EXT_SOURCE_1", "AMT_INCOME_TOTAL", "BURO_SUM"],
        top_n=2,
        output_path=fi_path,
    )

    fold_importance = pd.DataFrame(
        {
            "feature": ["EXT_SOURCE_1", "BURO_X", "AMT_CREDIT", "DAYS_BIRTH"],
            "lgb_gain": [5.0, 3.0, 2.0, 1.0],
            "cat_importance": [1.0, 2.0, 0.5, 0.2],
        }
    )
    efi_fig, efi_top = plot_ensemble_feature_importance(fold_importance, top_n=3, output_path=efi_path)

    assert roc_fig is not None and pr_fig is not None and fi_fig is not None and efi_fig is not None
    assert roc_path.exists() and pr_path.exists() and fi_path.exists() and efi_path.exists()
    assert len(fi_top) == 2
    assert set(["feature", "gain", "category"]).issubset(fi_top.columns)
    assert len(efi_top) == 3
    assert set(["feature", "ensemble_gain", "category"]).issubset(efi_top.columns)


def test_feature_bucket_and_importance_errors(tmp_path: Path):
    assert feature_bucket("BURO_DPD") == "Subsidiary aggregates"
    assert feature_bucket("EXT_SOURCE_2") == "External scores"
    assert feature_bucket("DAYS_BIRTH") == "Time-based"
    assert feature_bucket("CREDIT_INCOME_RATIO") == "Ratios"
    assert feature_bucket("AMT_CREDIT") == "Capacity"
    assert feature_bucket("X_UNKNOWN") == "Other"

    prepared = _prepare_importance_table(pd.Series([0.3, 0.2], index=["AMT_CREDIT", "EXT_SOURCE_2"]), top_n=1)
    assert list(prepared.columns) == ["feature", "gain", "category"]
    assert len(prepared) == 1

    try:
        plot_ensemble_feature_importance(pd.DataFrame(), output_path=tmp_path / "x.png")
        assert False, "Expected ValueError for empty fold_importance"
    except ValueError as exc:
        assert "empty" in str(exc)
