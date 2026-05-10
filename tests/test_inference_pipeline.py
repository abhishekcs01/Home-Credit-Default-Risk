from pathlib import Path

import numpy as np
import pandas as pd

from src.inference import predict_with_ensemble
from src.preprocessing import DesignMatrixPreprocessor
from src.utils import load_pickle, save_pickle


class DummyLGBModel:
    def predict(self, x, num_iteration=None):
        return np.clip(x[:, 0] / (np.abs(x[:, 0]).max() + 1e-6), 0, 1)

    @property
    def best_iteration(self):
        return 1


def _train_test_frames():
    rng = np.random.default_rng(123)
    n = 30
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(1000, 1000 + n),
            "TARGET": rng.integers(0, 2, size=n),
            "AMT_INCOME_TOTAL": rng.normal(100000, 15000, size=n),
            "AMT_CREDIT": rng.normal(300000, 50000, size=n),
            "ORGANIZATION_TYPE": [f"ORG_{i % 10}" for i in range(n)],
        }
    )
    return df.iloc[:20].copy(), df.iloc[20:].drop(columns=["TARGET"]).copy()


def test_predict_with_fold_bundle_and_pickle_roundtrip(tmp_path: Path):
    train_df, test_df = _train_test_frames()
    prep = DesignMatrixPreprocessor(ohe_max_categories=5).fit(train_df)
    fold_bundle = {
        "fold_models": [
            {
                "preprocessor": prep,
                "feature_keep_indices": None,
                "lgb_model": DummyLGBModel(),
                "cat_model": None,
            }
        ],
        "blend_weights": {"lgb": 1.0, "cat": 0.0},
    }
    pkl_path = tmp_path / "bundle.pkl"
    save_pickle(fold_bundle, pkl_path)
    loaded = load_pickle(pkl_path)
    preds = predict_with_ensemble(loaded, test_df)
    assert list(preds.columns) == ["SK_ID_CURR", "TARGET"]
    assert len(preds) == len(test_df)
    assert np.isfinite(preds["TARGET"]).all()
