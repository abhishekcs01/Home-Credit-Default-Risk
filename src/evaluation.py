import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from src import config
from src.utils import ensure_dir


def plot_roc_curve(y_true, y_score, output_path=config.ROC_CURVE_PATH, model_name: str = "OOF blended ensemble"):
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=model_name)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    return fig


def plot_pr_curve(y_true, y_score, output_path=config.PR_CURVE_PATH, model_name: str = "OOF blended ensemble"):
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=model_name)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    return fig


def feature_bucket(name: str) -> str:
    s = str(name).upper()
    if any(p in s for p in ("BURO_", "PREV_", "INST_", "POS_", "CC_", "BB_")):
        return "Subsidiary aggregates"
    if "EXT_SOURCE" in s or s.startswith("EXT_"):
        return "External scores"
    if any(k in s for k in ("AGE_YEARS", "EMPLOYMENT_", "DAYS_BIRTH", "DAYS_EMPLOYED")):
        return "Time-based"
    if any(k in s for k in ("RATIO", "CREDIT_INCOME", "ANNUITY_INCOME", "ANNUITY_CREDIT", "CHILDREN_RATIO")):
        return "Ratios"
    if any(k in s for k in ("AMT_", "INCOME", "ANNUITY", "GOODS_PRICE", "CREDIT")) and "RATIO" not in s:
        return "Capacity"
    return "Other"


def _prepare_importance_table(importance: pd.Series, top_n: int) -> pd.DataFrame:
    top = importance.sort_values(ascending=False).head(top_n).rename("gain").reset_index().rename(columns={"index": "feature"})
    top["category"] = top["feature"].map(feature_bucket)
    return top


def plot_feature_importance(model, feature_names: list[str], top_n: int = 15, output_path=config.FEATURE_IMPORTANCE_PATH):
    ensure_dir(output_path.parent)
    importance = pd.Series(model.feature_importance(importance_type="gain"), index=feature_names).sort_values(ascending=False)
    top = _prepare_importance_table(importance, top_n=top_n)
    plot_df = top.iloc[::-1].reset_index(drop=True)
    palette = {
        "Subsidiary aggregates": "#1B4332",
        "External scores": "#2E86AB",
        "Ratios": "#A23B72",
        "Time-based": "#F18F01",
        "Capacity": "#6A994E",
        "Other": "#888888",
    }
    colors = plot_df["category"].map(lambda x: palette.get(x, "#888888"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(plot_df["feature"], plot_df["gain"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Gain")
    ax.set_title("Top Feature Importance (LightGBM)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    return fig, top


def plot_ensemble_feature_importance(
    fold_importance: pd.DataFrame,
    *,
    top_n: int = 15,
    output_path=config.FEATURE_IMPORTANCE_PATH,
) -> tuple[plt.Figure, pd.DataFrame]:
    ensure_dir(output_path.parent)
    if fold_importance.empty:
        raise ValueError("fold_importance table is empty.")
    grouped = (
        fold_importance.assign(ensemble_gain=0.7 * fold_importance["lgb_gain"] + 0.3 * fold_importance["cat_importance"])
        .groupby("feature", as_index=True)["ensemble_gain"]
        .mean()
    )
    top = _prepare_importance_table(grouped, top_n=top_n).rename(columns={"gain": "ensemble_gain"})
    plot_df = top.iloc[::-1].reset_index(drop=True)

    palette = {
        "Subsidiary aggregates": "#1B4332",
        "External scores": "#2E86AB",
        "Ratios": "#A23B72",
        "Time-based": "#F18F01",
        "Capacity": "#6A994E",
        "Other": "#888888",
    }
    colors = plot_df["category"].map(lambda x: palette.get(x, "#888888"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(plot_df["feature"], plot_df["ensemble_gain"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Ensemble Gain")
    ax.set_title("Top Feature Importance (Ensemble)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    return fig, top

