from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance

from src.utils import get_logger

LOGGER = get_logger("feature_selection")


@dataclass
class FeatureRankingReport:
    feature_names: list[str]
    lgb_gain: dict[str, float] = field(default_factory=dict)
    mutual_info: dict[str, float] = field(default_factory=dict)
    permutation: dict[str, float] = field(default_factory=dict)
    shap_mean_abs: dict[str, float] = field(default_factory=dict)
    composite_score: dict[str, float] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name in self.feature_names:
            rows.append(
                {
                    "feature": name,
                    "lgb_gain": self.lgb_gain.get(name, 0.0),
                    "mutual_info": self.mutual_info.get(name, 0.0),
                    "permutation": self.permutation.get(name, 0.0),
                    "shap_mean_abs": self.shap_mean_abs.get(name, 0.0),
                    "composite_score": self.composite_score.get(name, 0.0),
                    "selected": name in self.selected_features,
                }
            )
        return pd.DataFrame(rows).sort_values("composite_score", ascending=False)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rank_features(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    lgb_model=None,
    sklearn_model=None,
    shap_summary: dict[str, float] | None = None,
    prune_bottom_frac: float = 0.06,
    max_samples_mi: int = 20_000,
    random_state: int = 42,
) -> FeatureRankingReport:
    n_features = x.shape[1]
    names = feature_names if len(feature_names) == n_features else [f"f_{i}" for i in range(n_features)]

    lgb_gain: dict[str, float] = {}
    if lgb_model is not None and hasattr(lgb_model, "feature_importance"):
        imp = np.asarray(lgb_model.feature_importance(importance_type="gain"), dtype=float)
        lgb_gain = {names[i]: float(imp[i]) for i in range(min(len(names), len(imp)))}

    mi_scores: dict[str, float] = {}
    if x.size:
        n = min(len(y), max_samples_mi)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=n, replace=False) if len(y) > n else np.arange(len(y))
        x_sub = x[idx]
        y_sub = y[idx]
        try:
            mi = mutual_info_classif(x_sub, y_sub, random_state=random_state, discrete_features=False)
            mi_scores = {names[i]: float(mi[i]) for i in range(min(len(names), len(mi)))}
        except Exception as exc:
            LOGGER.warning("Mutual information scoring failed: %s", exc)

    perm_scores: dict[str, float] = {}
    if sklearn_model is not None and x.size:
        try:
            n_perm = min(len(y), 5_000)
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(y), size=n_perm, replace=False) if len(y) > n_perm else np.arange(len(y))
            result = permutation_importance(
                sklearn_model,
                x[idx],
                y[idx],
                n_repeats=3,
                random_state=random_state,
                scoring="roc_auc",
                n_jobs=1,
            )
            perm_scores = {names[i]: float(result.importances_mean[i]) for i in range(min(len(names), len(result.importances_mean)))}
        except Exception as exc:
            LOGGER.warning("Permutation importance failed: %s", exc)

    shap_scores: dict[str, float] = {}
    if shap_summary:
        for name, value in shap_summary.items():
            if isinstance(value, dict):
                shap_scores[name] = float(value.get("mean_abs", 0.0))
            else:
                shap_scores[name] = float(value)
    norm_lgb = _normalize_scores(lgb_gain)
    norm_mi = _normalize_scores(mi_scores)
    norm_perm = _normalize_scores(perm_scores)
    norm_shap = _normalize_scores(shap_scores)

    composite: dict[str, float] = {}
    for name in names:
        composite[name] = (
            0.35 * norm_lgb.get(name, 0.0)
            + 0.25 * norm_mi.get(name, 0.0)
            + 0.20 * norm_perm.get(name, 0.0)
            + 0.20 * norm_shap.get(name, 0.0)
        )

    selected = list(names)
    if prune_bottom_frac > 0 and len(names) >= 80:
        ordered = sorted(composite.items(), key=lambda kv: kv[1])
        n_drop = max(1, int(len(ordered) * prune_bottom_frac))
        selected = [k for k, _ in ordered[n_drop:]]
        if len(selected) < max(60, int(0.42 * len(names))):
            selected = list(names)

    return FeatureRankingReport(
        feature_names=names,
        lgb_gain=lgb_gain,
        mutual_info=mi_scores,
        permutation=perm_scores,
        shap_mean_abs=shap_scores,
        composite_score=composite,
        selected_features=selected,
    )
