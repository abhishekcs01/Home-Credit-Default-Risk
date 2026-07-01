from __future__ import annotations

import pandas as pd

from src.data import load_data as data_loading
from src.data.synthetic_data import raw_data_available, using_synthetic_data


def test_synthetic_fallback_when_raw_data_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loading.config, "APPLICATION_TRAIN_PATH", tmp_path / "missing_train.csv")
    monkeypatch.setattr(data_loading.config, "APPLICATION_TEST_PATH", tmp_path / "missing_test.csv")

    assert raw_data_available() is False
    assert using_synthetic_data() is True

    train = data_loading.load_application_train(optimize_memory=False)
    test = data_loading.load_application_test(optimize_memory=False)
    aux = data_loading.load_auxiliary_tables(optimize_memory=False, sk_ids=train["SK_ID_CURR"])

    assert "TARGET" in train.columns
    assert "TARGET" not in test.columns
    assert len(train) >= 100
    assert len(test) >= 100
    assert set(aux) == {
        "bureau",
        "bureau_balance",
        "previous_application",
        "installments_payments",
        "pos_cash_balance",
        "credit_card_balance",
    }
    assert not aux["bureau"].empty
    assert aux["bureau"]["SK_ID_CURR"].isin(train["SK_ID_CURR"]).all()


def test_preprocess_builds_merged_artifacts_from_synthetic(monkeypatch, tmp_path):
    from scripts import preprocess_data as pp

    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(pp.config, "PROCESSED_DATA_DIR", processed)
    monkeypatch.setattr(pp.config, "MERGED_TRAIN_PATH", processed / "merged_train.pkl")
    monkeypatch.setattr(pp.config, "MERGED_TEST_PATH", processed / "merged_test.pkl")
    monkeypatch.setattr("src.data.synthetic_data.DEFAULT_SYNTHETIC_TRAIN_ROWS", 120)
    monkeypatch.setattr("src.data.synthetic_data.DEFAULT_SYNTHETIC_TEST_ROWS", 60)

    train_path, test_path = pp.run_preprocessing(use_cache=False, optimize_memory=False)

    assert train_path == processed / "merged_train.pkl"
    assert test_path == processed / "merged_test.pkl"
    merged_train = pd.read_pickle(train_path)
    merged_test = pd.read_pickle(test_path)
    assert len(merged_train) == 120
    assert len(merged_test) == 60
    assert "TARGET" in merged_train.columns
    assert "TARGET" not in merged_test.columns
    assert merged_train.shape[1] > 50


def test_ensure_model_artifact_bootstraps_via_pipeline(monkeypatch, tmp_path):
    from scripts import api_runtime

    bundle_path = tmp_path / "ensemble_model.pkl"
    monkeypatch.setattr(api_runtime.config, "MODEL_BUNDLE_PATH", bundle_path)

    called: list[bool] = []

    def _fake_ensure_model_bundle(**kwargs):
        called.append(True)
        bundle_path.write_bytes(b"demo-bundle")
        return bundle_path

    monkeypatch.setattr("src.data.pipeline.ensure_model_bundle", _fake_ensure_model_bundle)

    api_runtime.ensure_model_artifact()

    assert called == [True]
    assert bundle_path.is_file()


def test_pipeline_helpers_short_circuit_when_artifacts_exist(monkeypatch, tmp_path):
    from src.data import pipeline

    train_pkl = tmp_path / "merged_train.pkl"
    test_pkl = tmp_path / "merged_test.pkl"
    bundle_pkl = tmp_path / "ensemble_model.pkl"
    train_pkl.write_bytes(b"train")
    test_pkl.write_bytes(b"test")
    bundle_pkl.write_bytes(b"bundle")

    monkeypatch.setattr(pipeline.config, "MERGED_TRAIN_PATH", train_pkl)
    monkeypatch.setattr(pipeline.config, "MERGED_TEST_PATH", test_pkl)
    monkeypatch.setattr(pipeline.config, "MODEL_BUNDLE_PATH", bundle_pkl)

    assert pipeline.ensure_processed_datasets() == (train_pkl, test_pkl)
    assert pipeline.ensure_model_bundle() == bundle_pkl


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
