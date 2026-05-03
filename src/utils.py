import logging
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

_LOGGING_INITIALIZED = False


def init_logging(log_file: Path | None = None) -> None:
    """Configure the ``home_credit`` logger once (stdout; optional pipeline log file)."""
    global _LOGGING_INITIALIZED
    base = logging.getLogger("home_credit")
    base.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")

    if not _LOGGING_INITIALIZED:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        base.addHandler(ch)
        _LOGGING_INITIALIZED = True

    if log_file is not None:
        log_resolved = log_file.resolve()
        for h in base.handlers:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_resolved):
                break
        else:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            base.addHandler(fh)


def get_logger(name: str = "app") -> logging.Logger:
    init_logging()
    return logging.getLogger(f"home_credit.{name}")


@contextmanager
def timer(name: str):
    log = get_logger("timing")
    start = time.time()
    log.info("Starting: %s", name)
    yield
    elapsed = time.time() - start
    log.info("%s finished in %.2fs", name, elapsed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def reduce_mem_usage(frame: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mb = frame.memory_usage(deep=True).sum() / 1024**2
    out = frame.copy(deep=False)
    for col in out.columns:
        col_type = out[col].dtype
        if col_type == "object":
            try:
                out[col] = out[col].astype("category")
            except (TypeError, ValueError):
                pass
        elif str(col_type).startswith("int"):
            lo, hi = out[col].min(), out[col].max()
            if lo >= np.iinfo(np.int8).min and hi <= np.iinfo(np.int8).max:
                out[col] = out[col].astype(np.int8)
            elif lo >= np.iinfo(np.int16).min and hi <= np.iinfo(np.int16).max:
                out[col] = out[col].astype(np.int16)
            elif lo >= np.iinfo(np.int32).min and hi <= np.iinfo(np.int32).max:
                out[col] = out[col].astype(np.int32)
        elif str(col_type).startswith("float"):
            out[col] = pd.to_numeric(out[col], downcast="float")

    end_mb = out.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        get_logger().info(
            "Memory usage reduced from %.1f MB to %.1f MB (%.1f%% reduction)",
            start_mb,
            end_mb,
            100 * (start_mb - end_mb) / start_mb if start_mb else 0.0,
        )
    return out

