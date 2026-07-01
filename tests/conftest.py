from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import DesignMatrixPreprocessor
from src.utils import save_pickle
from tests.fixtures.synthetic_generators import make_application_frame


class DummyLGBModel:
    def predict(self, x, num_iteration=None):
        scaled = x[:, 0] / (np.abs(x[:, 0]).max() + 1e-6)
        return np.clip(scaled, 0.0, 1.0)

    @property
    def best_iteration(self):
        return 1


@pytest.fixture(autouse=True)
def deterministic_seed():
    random.seed(42)
    np.random.seed(42)


@pytest.fixture(autouse=True)
def training_device_cpu_for_tests(monkeypatch):
    """Keep pytest deterministic and independent of workstation GPU auto-resolution."""
    monkeypatch.setenv("TRAINING_DEVICE", "cpu")
    from src.config.settings import load_config

    load_config.cache_clear()
    yield
    load_config.cache_clear()


@pytest.fixture(autouse=True)
def isolate_tests_from_kaggle_data(monkeypatch):
    """Tests never require Kaggle CSVs or local raw data on disk."""
    monkeypatch.setattr("src.data.synthetic_data.raw_data_available", lambda: False)
    monkeypatch.setattr("src.data.load_data.raw_data_available", lambda: False)


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    return make_application_frame(n_rows=140, seed=111, include_target=True)


@pytest.fixture
def synthetic_inference_df() -> pd.DataFrame:
    return make_application_frame(n_rows=35, seed=222, include_target=False)


@pytest.fixture
def data_factory() -> Callable[..., pd.DataFrame]:
    return make_application_frame


@pytest.fixture
def bundle_path(tmp_path: Path, synthetic_train_df: pd.DataFrame) -> Path:
    preprocessor = DesignMatrixPreprocessor(ohe_max_categories=5).fit(synthetic_train_df)
    bundle = {
        "fold_models": [
            {
                "preprocessor": preprocessor,
                "feature_keep_indices": None,
                "lgb_model": DummyLGBModel(),
                "cat_model": None,
            }
        ],
        "blend_weights": {"lgb": 1.0, "cat": 0.0},
    }
    path = tmp_path / "ensemble_bundle.pkl"
    save_pickle(bundle, path)
    return path
