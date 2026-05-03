import numpy as np
import pandas as pd

from src.model_training import evaluate_models


def test_evaluate_models_runs_on_synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": rng.integers(0, 2, size=n),
            "AMT_INCOME_TOTAL": rng.normal(150000, 30000, size=n),
            "AMT_CREDIT": rng.normal(500000, 100000, size=n),
            "AMT_ANNUITY": rng.normal(25000, 5000, size=n),
            "EXT_SOURCE_1": rng.uniform(0, 1, size=n),
            "EXT_SOURCE_2": rng.uniform(0, 1, size=n),
            "EXT_SOURCE_3": rng.uniform(0, 1, size=n),
            "CAT_COL": rng.choice(["A", "B", "C"], size=n),
        }
    )
    results = evaluate_models(df, test_size=0.25, prune_bottom_frac=0.0)
    assert "auc_lgb" in results and 0.0 <= results["auc_lgb"] <= 1.0
    assert "feature_names" in results and len(results["feature_names"]) > 0

