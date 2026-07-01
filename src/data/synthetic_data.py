from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

DEFAULT_SYNTHETIC_TRAIN_ROWS = 800
DEFAULT_SYNTHETIC_TEST_ROWS = 300
DEFAULT_SYNTHETIC_SEED = 42


def raw_data_available() -> bool:
    """Return True when the primary Kaggle training file is present on disk."""
    return config.APPLICATION_TRAIN_PATH.is_file()


def using_synthetic_data() -> bool:
    """Return True when loaders will synthesize tables instead of reading CSVs."""
    return not raw_data_available()


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_application_frame(
    *,
    n_rows: int,
    seed: int,
    include_target: bool,
    id_start: int = 100_000,
) -> pd.DataFrame:
    rng = _rng(seed)
    sk_ids = np.arange(id_start, id_start + n_rows, dtype=np.int64)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": sk_ids,
            "AMT_INCOME_TOTAL": rng.normal(160_000, 25_000, size=n_rows),
            "AMT_CREDIT": rng.normal(540_000, 90_000, size=n_rows),
            "AMT_ANNUITY": rng.normal(26_000, 4_500, size=n_rows),
            "AMT_GOODS_PRICE": rng.normal(450_000, 80_000, size=n_rows),
            "CNT_CHILDREN": rng.integers(0, 4, size=n_rows),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, size=n_rows),
            "DAYS_BIRTH": -rng.integers(7_500, 25_000, size=n_rows),
            "DAYS_EMPLOYED": -rng.integers(30, 12_000, size=n_rows),
            "EXT_SOURCE_1": rng.uniform(0.0, 1.0, size=n_rows),
            "EXT_SOURCE_2": rng.uniform(0.0, 1.0, size=n_rows),
            "EXT_SOURCE_3": rng.uniform(0.0, 1.0, size=n_rows),
            "ORGANIZATION_TYPE": [f"ORG_{i % 18}" for i in range(n_rows)],
            "OCCUPATION_TYPE": rng.choice(
                ["Laborers", "Managers", "Core staff", "Sales staff", "Unknown"],
                size=n_rows,
            ),
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Commercial associate", "Pensioner", "State servant"],
                size=n_rows,
            ),
            "NAME_TYPE_SUITE": rng.choice(["Unaccompanied", "Family", "Spouse, partner"], size=n_rows),
            "CODE_GENDER": rng.choice(["M", "F"], size=n_rows),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], size=n_rows),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], size=n_rows),
        }
    )

    df.loc[df.index[::10], "EXT_SOURCE_1"] = np.nan
    df.loc[df.index[::14], "AMT_ANNUITY"] = np.nan
    df.loc[df.index[::17], "OCCUPATION_TYPE"] = None

    if include_target:
        logits = (
            -1.5
            + 2.0 * (1.0 - df["EXT_SOURCE_2"].fillna(0.5))
            + 0.0000015 * (df["AMT_CREDIT"] - df["AMT_INCOME_TOTAL"])
        )
        prob = 1.0 / (1.0 + np.exp(-logits))
        df["TARGET"] = (rng.uniform(0.0, 1.0, size=n_rows) < prob).astype(int)

    return df


def _make_bureau(sk_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bureau_id = 1
    for sk_id in sk_ids:
        n_accounts = int(rng.integers(1, 4))
        for _ in range(n_accounts):
            rows.append(
                {
                    "SK_ID_BUREAU": bureau_id,
                    "SK_ID_CURR": int(sk_id),
                    "DAYS_CREDIT": int(-rng.integers(10, 2_500)),
                    "AMT_CREDIT_SUM": float(rng.uniform(500.0, 8_000.0)),
                    "CREDIT_ACTIVE": rng.choice(["Active", "Closed"]),
                    "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Mortgage"]),
                }
            )
            bureau_id += 1
    return pd.DataFrame(rows)


def _make_bureau_balance(bureau: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sk_bureau in bureau["SK_ID_BUREAU"].unique():
        n_months = int(rng.integers(2, 6))
        for month in range(1, n_months + 1):
            rows.append(
                {
                    "SK_ID_BUREAU": int(sk_bureau),
                    "MONTHS_BALANCE": -month,
                    "STATUS": str(int(rng.integers(0, 4))),
                }
            )
    return pd.DataFrame(rows)


def _make_previous_application(sk_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prev_id = 1
    for sk_id in sk_ids:
        n_prev = int(rng.integers(1, 4))
        for _ in range(n_prev):
            rows.append(
                {
                    "SK_ID_PREV": prev_id,
                    "SK_ID_CURR": int(sk_id),
                    "DAYS_DECISION": int(-rng.integers(5, 900)),
                    "AMT_APPLICATION": float(rng.uniform(2_000.0, 20_000.0)),
                    "AMT_CREDIT": float(rng.uniform(1_500.0, 18_000.0)),
                    "AMT_ANNUITY": float(rng.uniform(200.0, 2_500.0)),
                    "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"]),
                    "FLAG_LAST_APPL_PER_CONTRACT": rng.choice(["Y", "N"]),
                }
            )
            prev_id += 1
    return pd.DataFrame(rows)


def _make_installments(previous: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in previous.itertuples(index=False):
        n_payments = int(rng.integers(2, 5))
        for _ in range(n_payments):
            days_inst = int(-rng.integers(20, 700))
            rows.append(
                {
                    "SK_ID_CURR": int(row.SK_ID_CURR),
                    "SK_ID_PREV": int(row.SK_ID_PREV),
                    "DAYS_ENTRY_PAYMENT": days_inst + int(rng.integers(-10, 25)),
                    "DAYS_INSTALMENT": days_inst,
                    "AMT_PAYMENT": float(rng.uniform(40.0, 250.0)),
                    "AMT_INSTALMENT": float(rng.uniform(50.0, 260.0)),
                }
            )
    return pd.DataFrame(rows)


def _make_pos_cash(previous: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in previous.itertuples(index=False):
        n_months = int(rng.integers(2, 5))
        for month in range(1, n_months + 1):
            rows.append(
                {
                    "SK_ID_CURR": int(row.SK_ID_CURR),
                    "SK_ID_PREV": int(row.SK_ID_PREV),
                    "MONTHS_BALANCE": -month,
                    "SK_DPD": int(rng.integers(0, 8)),
                    "CNT_INSTALMENT_FUTURE": int(rng.integers(0, 6)),
                }
            )
    return pd.DataFrame(rows)


def _make_credit_card(previous: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in previous.itertuples(index=False):
        n_months = int(rng.integers(2, 5))
        limit = float(rng.uniform(500.0, 5_000.0))
        for month in range(1, n_months + 1):
            rows.append(
                {
                    "SK_ID_CURR": int(row.SK_ID_CURR),
                    "SK_ID_PREV": int(row.SK_ID_PREV),
                    "MONTHS_BALANCE": -month,
                    "AMT_BALANCE": float(rng.uniform(0.0, limit)),
                    "AMT_CREDIT_LIMIT_ACTUAL": limit,
                    "SK_DPD": int(rng.integers(0, 12)),
                }
            )
    return pd.DataFrame(rows)


def generate_synthetic_auxiliary_tables_consistent(
    sk_ids: np.ndarray,
    *,
    seed: int = DEFAULT_SYNTHETIC_SEED,
) -> dict[str, pd.DataFrame]:
    """Generate auxiliary tables with a single shared previous-application backbone."""
    rng = _rng(seed + 1)
    bureau = _make_bureau(sk_ids, rng)
    previous = _make_previous_application(sk_ids, rng)
    return {
        "bureau": bureau,
        "bureau_balance": _make_bureau_balance(bureau, rng),
        "previous_application": previous,
        "installments_payments": _make_installments(previous, rng),
        "pos_cash_balance": _make_pos_cash(previous, rng),
        "credit_card_balance": _make_credit_card(previous, rng),
    }


def generate_synthetic_application_train(
    *,
    n_rows: int | None = None,
    seed: int = DEFAULT_SYNTHETIC_SEED,
) -> pd.DataFrame:
    return make_application_frame(
        n_rows=n_rows or DEFAULT_SYNTHETIC_TRAIN_ROWS,
        seed=seed,
        include_target=True,
    )


def generate_synthetic_application_test(
    *,
    n_rows: int | None = None,
    seed: int = DEFAULT_SYNTHETIC_SEED,
) -> pd.DataFrame:
    return make_application_frame(
        n_rows=n_rows or DEFAULT_SYNTHETIC_TEST_ROWS,
        seed=seed + 10_000,
        include_target=False,
        id_start=200_000,
    )
