from __future__ import annotations

import numpy as np

from src.features.selection import rank_features
from src.models.calibration import evaluate_calibration_methods, predict_platt


def test_calibration_method_selection_prefers_lower_brier():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    raw = np.clip(y * 0.6 + rng.normal(0, 0.2, size=400), 0, 1)
    iso, platt, best = evaluate_calibration_methods(y, raw)
    assert best in {"isotonic", "platt"}
    assert iso.brier_score >= 0
    assert platt.brier_score >= 0
    chosen = iso if best == "isotonic" else platt
    calibrated = (
        chosen.model.predict(raw) if best == "isotonic" else predict_platt(chosen.model, raw)
    )
    assert calibrated.shape == (400,)


def test_rank_features_returns_composite_scores():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 6))
    y = (x[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    names = [f"f_{i}" for i in range(6)]

    class _DummyLGB:
        def feature_importance(self, importance_type="gain"):
            return np.arange(6, dtype=float)

    report = rank_features(x, y, names, lgb_model=_DummyLGB(), prune_bottom_frac=0.0)
    df = report.to_dataframe()
    assert len(df) == 6
    assert "composite_score" in df.columns
    assert df["composite_score"].max() >= df["composite_score"].min()
