import numpy as np
import pandas as pd


def predict_with_ensemble(model_bundle: dict, merged_test: pd.DataFrame) -> pd.DataFrame:
    prep = model_bundle["preprocessor"]
    models = model_bundle["models"]
    keep_idx = model_bundle.get("feature_keep_indices")

    x_test = prep.transform(merged_test)
    if keep_idx is not None:
        x_test = x_test[:, keep_idx]

    ensemble_test_preds = np.column_stack([m.predict(x_test, num_iteration=m.best_iteration) for m in models])
    predictions = ensemble_test_preds.mean(axis=1)
    return pd.DataFrame({"SK_ID_CURR": merged_test["SK_ID_CURR"], "TARGET": predictions})

