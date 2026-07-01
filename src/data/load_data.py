from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import synthetic_data
from src.utils import get_logger, reduce_mem_usage

LOGGER = get_logger("data.load_data")


def raw_data_available() -> bool:
    return synthetic_data.raw_data_available()


def using_synthetic_data() -> bool:
    return synthetic_data.using_synthetic_data()


def load_application_train(optimize_memory: bool = True) -> pd.DataFrame:
    if raw_data_available():
        df = pd.read_csv(config.APPLICATION_TRAIN_PATH)
    else:
        LOGGER.warning(
            "Raw training data not found at %s; using synthetic sample data.",
            config.APPLICATION_TRAIN_PATH,
        )
        df = synthetic_data.generate_synthetic_application_train()
    return reduce_mem_usage(df) if optimize_memory else df


def load_application_test(optimize_memory: bool = True) -> pd.DataFrame:
    if raw_data_available() and config.APPLICATION_TEST_PATH.is_file():
        df = pd.read_csv(config.APPLICATION_TEST_PATH)
    else:
        LOGGER.warning(
            "Raw test data not found at %s; using synthetic sample data.",
            config.APPLICATION_TEST_PATH,
        )
        df = synthetic_data.generate_synthetic_application_test()
    return reduce_mem_usage(df) if optimize_memory else df


def load_auxiliary_tables(
    optimize_memory: bool = True,
    *,
    sk_ids: np.ndarray | pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    if raw_data_available():
        tables = {
            "bureau": pd.read_csv(config.BUREAU_PATH),
            "bureau_balance": pd.read_csv(config.BUREAU_BALANCE_PATH),
            "previous_application": pd.read_csv(config.PREVIOUS_APPLICATION_PATH),
            "installments_payments": pd.read_csv(config.INSTALLMENTS_PAYMENTS_PATH),
            "pos_cash_balance": pd.read_csv(config.POS_CASH_BALANCE_PATH),
            "credit_card_balance": pd.read_csv(config.CREDIT_CARD_BALANCE_PATH),
        }
    else:
        LOGGER.warning("Raw auxiliary tables missing; generating synthetic subsidiary tables.")
        if sk_ids is None:
            train_df = synthetic_data.generate_synthetic_application_train()
            sk_ids = train_df["SK_ID_CURR"].to_numpy()
        else:
            sk_ids = np.asarray(sk_ids)
        tables = synthetic_data.generate_synthetic_auxiliary_tables_consistent(sk_ids)
    if optimize_memory:
        return {name: reduce_mem_usage(df) for name, df in tables.items()}
    return tables


__all__ = [
    "load_application_test",
    "load_application_train",
    "load_auxiliary_tables",
    "raw_data_available",
    "using_synthetic_data",
]
