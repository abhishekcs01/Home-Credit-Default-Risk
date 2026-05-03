import pandas as pd

from src import config
from src.utils import reduce_mem_usage


def load_application_train(optimize_memory: bool = True) -> pd.DataFrame:
    df = pd.read_csv(config.APPLICATION_TRAIN_PATH)
    return reduce_mem_usage(df) if optimize_memory else df


def load_application_test(optimize_memory: bool = True) -> pd.DataFrame:
    df = pd.read_csv(config.APPLICATION_TEST_PATH)
    return reduce_mem_usage(df) if optimize_memory else df


def load_auxiliary_tables(optimize_memory: bool = True) -> dict[str, pd.DataFrame]:
    tables = {
        "bureau": pd.read_csv(config.BUREAU_PATH),
        "bureau_balance": pd.read_csv(config.BUREAU_BALANCE_PATH),
        "previous_application": pd.read_csv(config.PREVIOUS_APPLICATION_PATH),
        "installments_payments": pd.read_csv(config.INSTALLMENTS_PAYMENTS_PATH),
        "pos_cash_balance": pd.read_csv(config.POS_CASH_BALANCE_PATH),
        "credit_card_balance": pd.read_csv(config.CREDIT_CARD_BALANCE_PATH),
    }
    if optimize_memory:
        return {name: reduce_mem_usage(df) for name, df in tables.items()}
    return tables

