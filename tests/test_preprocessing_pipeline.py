import numpy as np
import pandas as pd

from src.preprocessing import DesignMatrixPreprocessor


def _synthetic_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "TARGET": rng.integers(0, 2, size=n),
            "AMT_INCOME_TOTAL": rng.normal(120000, 25000, size=n),
            "AMT_CREDIT": rng.normal(350000, 50000, size=n),
            "EXT_SOURCE_1": rng.uniform(0, 1, size=n),
            "ORGANIZATION_TYPE": [f"ORG_{i % 25}" for i in range(n)],
            "CAT_SMALL": rng.choice(["A", "B", "C"], size=n),
        }
    )


def test_preprocessor_target_encoding_and_alignment_are_consistent():
    df = _synthetic_df()
    train_df = df.iloc[:45].copy()
    val_df = df.iloc[45:].copy()
    val_df.loc[val_df.index[:4], "ORGANIZATION_TYPE"] = "ORG_NEVER_SEEN"
    val_df = val_df.drop(columns=["CAT_SMALL"])
    val_df["EXTRA_COL"] = 123

    prep = DesignMatrixPreprocessor(ohe_max_categories=5).fit(train_df)
    x_train = prep.transform(train_df)
    x_val = prep.transform(val_df)

    assert x_train.shape[1] == x_val.shape[1]
    assert x_train.shape[0] == len(train_df)
    assert x_val.shape[0] == len(val_df)
    assert np.isfinite(x_val).all()
    assert any(name.endswith("__TE") for name in prep.feature_names_)
