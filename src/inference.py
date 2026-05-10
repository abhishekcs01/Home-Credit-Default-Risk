import numpy as np
import pandas as pd

from src.utils import load_pickle


def predict_with_ensemble(model_bundle: dict, merged_test: pd.DataFrame) -> pd.DataFrame:
    fold_models = model_bundle.get("fold_models")
    if fold_models is None:
        # Backward-compatibility with older bundle format.
        prep = model_bundle["preprocessor"]
        models = model_bundle["models"]
        keep_idx = model_bundle.get("feature_keep_indices")
        x_test = prep.transform(merged_test)
        if keep_idx is not None:
            x_test = x_test[:, keep_idx]
        ensemble_test_preds = np.column_stack([m.predict(x_test, num_iteration=m.best_iteration) for m in models])
        predictions = ensemble_test_preds.mean(axis=1)
        return pd.DataFrame({"SK_ID_CURR": merged_test["SK_ID_CURR"], "TARGET": predictions})

    blend = model_bundle.get("blend_weights", {"lgb": 0.5, "cat": 0.5})
    w_lgb = float(blend.get("lgb", 0.5))
    w_cat = float(blend.get("cat", 0.5))
    fold_preds = []
    for fm in fold_models:
        prep = fm["preprocessor"]
        x_test = prep.transform(merged_test)
        keep_idx = fm.get("feature_keep_indices")
        if keep_idx is not None:
            x_test = x_test[:, keep_idx]
        lgb_model = fm["lgb_model"]
        lgb_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration)
        cat_model = fm.get("cat_model")
        if cat_model is not None:
            cat_pred = cat_model.predict_proba(x_test)[:, 1]
        else:
            cat_pred = np.zeros_like(lgb_pred)
        fold_preds.append(w_lgb * lgb_pred + w_cat * cat_pred)
    predictions = np.mean(np.column_stack(fold_preds), axis=1)
    return pd.DataFrame({"SK_ID_CURR": merged_test["SK_ID_CURR"], "TARGET": predictions})


class EnsembleInferenceEngine:
    """Lightweight helper for API/Locust integration."""

    def __init__(self, model_bundle_path):
        self.model_bundle_path = model_bundle_path
        self._bundle = None

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = load_pickle(self.model_bundle_path)
        return self._bundle

    def predict_dataframe(self, features: pd.DataFrame) -> pd.DataFrame:
        return predict_with_ensemble(self.bundle, features)

