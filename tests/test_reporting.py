from __future__ import annotations

import pandas as pd

from src.reporting.generate_reports import (
    generate_all_reports,
    generate_business_insights,
    generate_calibration_report,
    generate_executive_summary,
    generate_feature_report,
    generate_model_card,
    generate_model_comparison_report,
    generate_shap_report,
    generate_training_summary,
)


def test_generate_all_reports_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.reporting.generate_reports.config.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr("src.reporting.generate_reports.config.METRICS_DIR", tmp_path / "metrics")

    training_output = {
        "fold_metrics": [{"fold": 1, "auc_blend": 0.79, "n_features": 10}],
        "shap_importance_top": {"EXT_SOURCE_2": 0.5, "AMT_CREDIT": 0.3},
    }
    metrics = {
        "oof_auc_blend": 0.79,
        "oof_auc_calibrated": 0.795,
        "oof_auc_lgb": 0.788,
        "oof_auc_cat": 0.787,
        "oof_auc_stack": 0.792,
        "n_features_final": 10,
        "blend_weights": {"lgb": 0.5, "cat": 0.5, "xgb": 0.0},
    }
    ranking = pd.DataFrame({"feature": ["EXT_SOURCE_MEAN", "AMT_CREDIT"], "composite_score": [0.9, 0.1]})
    paths = generate_all_reports(
        training_output,
        metrics,
        calibration_report={"isotonic_brier": 0.1, "platt_brier": 0.11, "isotonic_auc": 0.79, "platt_auc": 0.79, "best_method": "isotonic"},
        feature_ranking=ranking,
        bundle_meta={"version": "1.0.0", "schema_version": "1.0", "trained_at": "t", "feature_count": 10, "oof_auc_calibrated": 0.795},
    )
    assert len(paths) >= 12
    assert (tmp_path / "reports" / "index.json").is_file()
    business_text = (tmp_path / "reports" / "business_insights.md").read_text(encoding="utf-8")
    assert "elevated default risk when positively associated" not in business_text
    assert "lower default rate" in business_text or "lower default" in business_text


def test_individual_report_generators(tmp_path, monkeypatch):
    monkeypatch.setattr("src.reporting.generate_reports.config.REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr("src.reporting.generate_reports.config.METRICS_DIR", tmp_path / "metrics")

    metrics = {"oof_auc_blend": 0.79, "oof_auc_calibrated": 0.795, "oof_auc_lgb": 0.788, "oof_auc_cat": 0.787, "oof_auc_stack": 0.792, "n_features_final": 5}
    generate_model_comparison_report({}, metrics)
    generate_calibration_report({"isotonic_brier": 0.1, "platt_brier": 0.12, "isotonic_auc": 0.79, "platt_auc": 0.78, "best_method": "platt"})
    generate_feature_report(pd.DataFrame({"feature": ["x"], "composite_score": [1.0]}))
    generate_shap_report({"x": 1.0})
    generate_shap_report(None)
    generate_training_summary({"fold_metrics": []}, metrics)
    generate_model_card({"version": "1", "schema_version": "1", "trained_at": "t", "feature_count": 1, "oof_auc_calibrated": 0.79})
    generate_business_insights(["EXT_SOURCE_MEAN"], metrics)
    assert (tmp_path / "reports" / "model_card.md").is_file()
    exec_md, _ = generate_executive_summary(metrics, ["EXT_SOURCE_MEAN"])
    assert exec_md.is_file()
    assert "Business Problem" in exec_md.read_text(encoding="utf-8")
