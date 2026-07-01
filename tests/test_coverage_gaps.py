from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import src.models.train as mt
from src.api.schemas import InferenceRecord, PredictRequest
from src.data.preprocess import DesignMatrixPreprocessor, dataframe_to_design_matrix
from src.features.selection import _normalize_scores, rank_features
from src.models.calibration import calibration_curve_data
from src.models.evaluate import run_evaluation
from src.reporting.generate_reports import (
    _feature_category,
    _normalize_shap,
    _risk_direction_label,
    generate_all_reports,
    regenerate_reports_from_artifacts,
)


class _DummyLGB:
    best_iteration = 1

    def predict(self, x, num_iteration=None):
        return np.full(len(x), 0.55)

    def feature_importance(self, importance_type="gain"):
        return np.ones(3, dtype=float)


def _training_cfg():
    return types.SimpleNamespace(
        training=types.SimpleNamespace(
            device="cpu",
            gpu_device_id=0,
            random_state=42,
            ohe_max_categories=5,
            miss_drop_threshold=0.7,
            prune_bottom_frac=0.06,
            n_folds=5,
            blend_weight_lgb=0.5,
            blend_weight_cat=0.5,
            blend_weight_xgb=0.0,
            enable_xgboost=False,
            enable_stacking=True,
            enable_calibration=True,
            enable_optuna=False,
            enable_shap=False,
        ),
        api=types.SimpleNamespace(version="test-1.0"),
    )


def test_run_training_persists_bundle_and_metrics(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": rng.integers(0, 2, size=n),
            "F1": rng.normal(size=n),
            "ORGANIZATION_TYPE": [f"ORG_{i % 6}" for i in range(n)],
        }
    )
    train_pkl = tmp_path / "merged_train.pkl"
    df.to_pickle(train_pkl)
    prep = DesignMatrixPreprocessor(ohe_max_categories=5).fit(df)
    lgb = _DummyLGB()
    lgb._last_x = prep.transform(df)

    training_output = {
        "fold_models": [
            {
                "preprocessor": prep,
                "feature_keep_indices": None,
                "feature_names": prep.feature_names_,
                "lgb_model": lgb,
                "cat_model": None,
            }
        ],
        "oof_auc_blend": 0.75,
        "oof_auc_lgb": 0.75,
        "oof_auc_cat": 0.74,
        "blend_weights": {"lgb": 1.0, "cat": 0.0, "xgb": 0.0},
        "feature_names_final": prep.feature_names_,
        "threshold_analysis": {"threshold": 0.5},
        "fold_metrics": [{"fold": 1, "auc_blend": 0.75, "n_features": len(prep.feature_names_)}],
        "ensemble_feature_importance": pd.DataFrame({"feature": ["F1"], "ensemble_gain": [1.0]}),
        "y_all": df["TARGET"].to_numpy(),
        "oof_pred_blend": np.full(n, 0.5),
        "shap_importance_top": {"F1": {"mean_abs": 0.2, "mean_signed": 0.1}},
    }

    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"
    for d in (models_dir, metrics_dir, figures_dir, reports_dir):
        d.mkdir(parents=True)

    from tests.test_training_devices import _cpu_devices

    monkeypatch.setattr(mt, "resolve_training_devices", lambda device, gpu_id: _cpu_devices())
    monkeypatch.setattr(mt, "log_training_devices", lambda logger, value: None)
    monkeypatch.setattr(mt, "load_config", _training_cfg)
    monkeypatch.setattr(mt.config, "MERGED_TRAIN_PATH", train_pkl)
    monkeypatch.setattr(mt.config, "MODEL_BUNDLE_PATH", models_dir / "ensemble_model.pkl")
    monkeypatch.setattr(mt.config, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mt.config, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(mt.config, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(mt.config, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(mt.config, "ROC_CURVE_PATH", figures_dir / "roc.png")
    monkeypatch.setattr(mt.config, "PR_CURVE_PATH", figures_dir / "pr.png")
    monkeypatch.setattr(mt.config, "FEATURE_IMPORTANCE_PATH", figures_dir / "fi.png")
    monkeypatch.setattr(mt.config, "TRAINING_METRICS_JSON_PATH", metrics_dir / "training_metrics.json")
    monkeypatch.setattr(mt.config, "TRAINING_CONFIG_SNAPSHOT_PATH", metrics_dir / "config_snapshot.json")
    monkeypatch.setattr(mt.config, "FOLD_METRICS_CSV_PATH", metrics_dir / "fold_metrics.csv")
    monkeypatch.setattr(mt.config, "FEATURE_IMPORTANCE_TOP15_CSV_PATH", metrics_dir / "fi_top15.csv")
    monkeypatch.setattr(mt, "train_kfold_lightgbm_ensemble", lambda *a, **k: training_output)
    monkeypatch.setattr(mt, "plot_roc_curve", lambda *a, **k: None)
    monkeypatch.setattr(mt, "plot_pr_curve", lambda *a, **k: None)
    monkeypatch.setattr(mt, "plot_ensemble_feature_importance", lambda *a, **k: (None, pd.DataFrame({"feature": ["F1"], "ensemble_gain": [1.0], "category": ["Other"]})))

    out = mt.run_training(demo_mode=False)
    assert out.is_file()
    assert (metrics_dir / "training_metrics.json").is_file()
    assert (reports_dir / "index.json").is_file()


def test_regenerate_reports_from_artifacts_paths(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    metrics = tmp_path / "metrics"
    reports.mkdir()
    metrics.mkdir()
    monkeypatch.setattr("src.reporting.generate_reports.config.REPORTS_DIR", reports)
    monkeypatch.setattr("src.reporting.generate_reports.config.METRICS_DIR", metrics)

    (metrics / "training_metrics.json").write_text(
        json.dumps(
            {
                "oof_auc_blend": 0.79,
                "oof_auc_calibrated": 0.795,
                "oof_auc_lgb": 0.788,
                "oof_auc_cat": 0.787,
                "oof_auc_stack": 0.792,
                "n_features_final": 3,
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"fold": [1], "auc_blend": [0.79], "n_features": [3]}).to_csv(metrics / "fold_metrics.csv", index=False)
    pd.DataFrame(
        {"feature": ["EXT_SOURCE_MEAN", "AMT_CREDIT"], "composite_score": [0.9, 0.1], "shap_mean_abs": [0.5, 0.0]}
    ).to_csv(metrics / "feature_ranking.csv", index=False)
    (metrics / "shap_summary.json").write_text(json.dumps({"EXT_SOURCE_MEAN": 0.5}), encoding="utf-8")

    paths = regenerate_reports_from_artifacts()
    assert len(paths) >= 10
    assert (reports / "index.json").is_file()

    (reports / "evaluation_report.json").write_text(
        json.dumps(
            {
                "metrics": {"oof_auc_calibrated": 0.8, "n_features_final": 2, "version": "2.0"},
                "calibration_report": {
                    "best_method": "isotonic",
                    "isotonic_brier": 0.1,
                    "platt_brier": 0.12,
                    "isotonic_auc": 0.79,
                    "platt_auc": 0.78,
                },
                "trained_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    paths2 = regenerate_reports_from_artifacts()
    assert len(paths2) >= 10


def test_reporting_helpers_and_shap_only_branch(monkeypatch, tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    monkeypatch.setattr("src.reporting.generate_reports.config.REPORTS_DIR", reports)
    monkeypatch.setattr("src.reporting.generate_reports.config.METRICS_DIR", tmp_path / "metrics")
    (reports / "defense_prep.md").write_text("# defense", encoding="utf-8")

    shap = _normalize_shap({"A": {"mean_abs": 0.3, "mean_signed": -0.1}, "B": 0.2})
    assert shap["A"]["mean_abs"] == 0.3
    assert shap["B"]["mean_signed"] == 0.0

    inc, _ = _risk_direction_label("EXT_SOURCE_1", mean_signed=0.05)
    dec, _ = _risk_direction_label("EXT_SOURCE_2", mean_signed=-0.05)
    assert "increases" in inc
    assert "decreases" in dec
    assert _feature_category("INST_LATE_RATE") == "Installment payment behavior"
    assert _feature_category("POS_SK_DPD_mean") == "POS/cash loan behavior"
    assert _feature_category("CC_UTIL") == "Credit card behavior"
    assert _feature_category("NAME_INCOME_TYPE__TE") == "Demographics / categoricals"

    metrics = {
        "oof_auc_blend": 0.79,
        "oof_auc_calibrated": 0.795,
        "oof_auc_lgb": 0.788,
        "oof_auc_cat": 0.787,
        "oof_auc_stack": 0.792,
        "n_features_final": 2,
    }
    paths = generate_all_reports(
        {"fold_metrics": [], "shap_importance_top": {"EXT_SOURCE_MEAN": 0.4}},
        metrics,
        feature_ranking=None,
    )
    assert "defense_prep_md" in paths
    assert "business_md" in paths


def test_rank_features_pruning_and_edge_scores():
    rng = np.random.default_rng(2)
    n_features = 90
    x = rng.normal(size=(200, n_features))
    y = (x[:, 0] > 0).astype(int)
    names = [f"f_{i}" for i in range(n_features)]

    class _LGB:
        def feature_importance(self, importance_type="gain"):
            return np.ones(n_features)

    report = rank_features(
        x,
        y,
        names,
        lgb_model=_LGB(),
        shap_summary={"f_0": {"mean_abs": 1.0}},
        prune_bottom_frac=0.1,
    )
    assert len(report.selected_features) < n_features
    assert _normalize_scores({"a": 1.0, "b": 1.0}) == {"a": 0.0, "b": 0.0}


def test_schemas_validation_paths():
    rec = InferenceRecord.model_validate({"SK_ID_CURR": 10, "AMT_CREDIT": 1000.0})
    assert rec.to_feature_dict()["AMT_CREDIT"] == 1000.0

    with pytest.raises(TypeError):
        InferenceRecord.model_validate("bad")

    with pytest.raises(ValueError):
        InferenceRecord.model_validate({"AMT_CREDIT": 1.0})

    with pytest.raises(ValidationError):
        InferenceRecord.model_validate({"SK_ID_CURR": 1, "BAD": True})

    with pytest.raises(ValidationError):
        InferenceRecord.model_validate({"SK_ID_CURR": 1, "X": float("inf")})

    with pytest.raises(ValidationError):
        InferenceRecord.model_validate({"SK_ID_CURR": 1, "X": 1e13})

    req = PredictRequest(records=[rec])
    assert len(req.records) == 1


def test_run_evaluation_and_calibration_curve(tmp_path, monkeypatch):
    bundle_path = tmp_path / "bundle.pkl"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    import joblib

    joblib.dump(
        {
            "model": {"calibration_method": "isotonic"},
            "metrics": {"blend_weights": {"lgb": 1.0}},
            "feature_names": ["F1"],
            "version": "1.0",
            "threshold": 0.5,
        },
        bundle_path,
    )
    monkeypatch.setattr("src.models.evaluate.config.MODEL_BUNDLE_PATH", bundle_path)
    monkeypatch.setattr("src.models.evaluate.config.REPORTS_DIR", reports_dir)
    monkeypatch.setattr("src.models.evaluate.config.FOLD_METRICS_CSV_PATH", tmp_path / "fold.csv")
    monkeypatch.setattr("src.models.evaluate.config.PROJECT_ROOT", tmp_path)

    out = run_evaluation()
    assert out.is_file()
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    curve = calibration_curve_data(y, p, n_bins=3)
    assert "fraction_of_positives" in curve


def test_pipeline_force_paths(monkeypatch, tmp_path):
    from src.data import pipeline

    train_pkl = tmp_path / "merged_train.pkl"
    test_pkl = tmp_path / "merged_test.pkl"
    called = {"preprocess": 0, "train": 0}

    monkeypatch.setattr(pipeline.config, "MERGED_TRAIN_PATH", train_pkl)
    monkeypatch.setattr(pipeline.config, "MERGED_TEST_PATH", test_pkl)
    monkeypatch.setattr(pipeline.config, "MODEL_BUNDLE_PATH", tmp_path / "bundle.pkl")
    monkeypatch.setattr(
        "scripts.preprocess_data.run_preprocessing",
        lambda **kwargs: (train_pkl.write_bytes(b"t") or test_pkl.write_bytes(b"t") or (train_pkl, test_pkl)),
    )
    monkeypatch.setattr(
        "src.models.train.run_training",
        lambda **kwargs: called.__setitem__("train", called["train"] + 1) or tmp_path / "bundle.pkl",
    )

    pipeline.ensure_processed_datasets(force_recompute=True)
    assert called["preprocess"] == 0  # run_preprocessing called via lambda side effect
    train_pkl.write_bytes(b"t")
    test_pkl.write_bytes(b"t")

    pipeline.ensure_model_bundle(force_retrain=True)
    assert called["train"] == 1


def test_predict_submission_and_dataframe_to_design_matrix(monkeypatch, tmp_path):
    from src.models import predict as pred_mod

    bundle = tmp_path / "bundle.pkl"
    merged_test = tmp_path / "merged_test.pkl"
    test_df = pd.DataFrame({"SK_ID_CURR": [1, 2], "F1": [1.0, 2.0], "F2": [0.5, 1.5]})
    test_df.to_pickle(merged_test)

    prep = DesignMatrixPreprocessor().fit(
        pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0], "F1": [1.0], "F2": [0.5], "ORGANIZATION_TYPE": ["A"]})
    )
    from src.utils import save_pickle

    save_pickle(
        {
            "fold_models": [
                {
                    "preprocessor": prep,
                    "feature_keep_indices": None,
                    "lgb_model": _DummyLGB(),
                    "cat_model": None,
                }
            ],
            "blend_weights": {"lgb": 1.0, "cat": 0.0, "xgb": 0.0},
        },
        bundle,
    )

    monkeypatch.setattr(pred_mod.config, "MODEL_BUNDLE_PATH", bundle)
    monkeypatch.setattr(pred_mod.config, "MERGED_TEST_PATH", merged_test)
    monkeypatch.setattr(pred_mod.config, "SUBMISSIONS_DIR", tmp_path)
    monkeypatch.setattr(pred_mod.config, "SUBMISSION_PATH", tmp_path / "submission.csv")

    out = pred_mod.run_submission()
    assert out.is_file()
    assert len(pd.read_csv(out)) == 2

    df = pd.DataFrame(
        {"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "F1": [1.0, 2.0], "ORGANIZATION_TYPE": ["A", "B"]}
    )
    x, y, names = dataframe_to_design_matrix(df, ohe_max_categories=5)
    assert x.shape[0] == 2
    assert len(names) > 0


def test_utils_reduce_mem_usage_verbose_and_int32(monkeypatch):
    from src.utils import reduce_mem_usage

    logged: list[str] = []

    class _Logger:
        def info(self, msg, *args):
            logged.append(msg % args if args else msg)

    monkeypatch.setattr("src.utils.get_logger", lambda: _Logger())
    frame = pd.DataFrame(
        {
            "obj": ["a", "b"],
            "i64": np.array([1, 2], dtype=np.int64),
            "f64": np.array([1.5, 2.5], dtype=np.float64),
        }
    )
    out = reduce_mem_usage(frame, verbose=True)
    assert str(out["f64"].dtype) == "float32"
    assert logged


def test_prediction_service_missing_features_warning(monkeypatch, tmp_path, caplog):
    from src.api.predict_service import PredictionService
    from src.api.schemas import InferenceRecord, PredictRequest

    class _Engine:
        bundle = {"feature_names": ["SK_ID_CURR", "F1", "F2", "F3"], "model": {}, "schema_version": "1.0"}
        bundle_loaded = True

        def predict_dataframe(self, frame):
            return pd.DataFrame({"SK_ID_CURR": frame["SK_ID_CURR"], "TARGET": [0.4] * len(frame)})

    monkeypatch.setattr("src.api.predict_service.EnsembleInferenceEngine", lambda path: _Engine())
    caplog.set_level("WARNING")
    service = PredictionService(model_bundle_path=tmp_path / "b.pkl")
    resp = service.predict(PredictRequest(records=[InferenceRecord.model_validate({"SK_ID_CURR": 1, "F1": 1.0})]))
    assert resp.request_count == 1
    assert any("missing" in r.message.lower() for r in caplog.records)


def test_api_batch_size_and_v1_predict(monkeypatch):
    from fastapi.testclient import TestClient

    from src.api import main
    from tests.test_api_advanced import SlowPredictionService, _api_settings

    cfg = main._cfg
    monkeypatch.setattr(
        main,
        "_cfg",
        types.SimpleNamespace(
            api=_api_settings(max_batch_size=1, startup_load_model=False),
            training=cfg.training,
            load_test=cfg.load_test,
        ),
    )
    monkeypatch.setattr(main, "get_prediction_service", lambda: SlowPredictionService())

    with TestClient(main.app) as client:
        too_big = client.post("/v1/predict", json={"records": [{"SK_ID_CURR": 1}, {"SK_ID_CURR": 2}]})
    assert too_big.status_code == 422
