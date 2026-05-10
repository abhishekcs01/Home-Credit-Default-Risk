from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.predict import sanitize_probability_series
from src.inference import EnsembleInferenceEngine


def test_sanitize_probability_series_finite_clip():
    s = pd.Series([np.nan, -0.1, 1.1, 0.5])
    out = sanitize_probability_series(s)
    assert pytest.approx(out.tolist()) == [0.5, 0.0, 1.0, 0.5]


def test_concurrent_predict_dataframe_completes(monkeypatch: pytest.MonkeyPatch):
    """Many concurrent /predict calls may overlap now (fold-parallel + per-cat locks); require no errors."""
    errors: list[BaseException] = []

    def fake_predict(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"], "TARGET": np.full(len(df), 0.5)})

    monkeypatch.setattr("src.inference.predict_with_ensemble", fake_predict)

    eng = EnsembleInferenceEngine(Path("dummy_bundle.pkl"))
    eng._bundle = {}
    df = pd.DataFrame({"SK_ID_CURR": [1]})

    def run() -> None:
        try:
            for _ in range(5):
                eng.predict_dataframe(df)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent predict failed: {errors!r}"
