from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from src.api.schemas import InferenceRecord
from src.config.settings import _read_yaml, load_config
from src.features.feature_schema import FeatureSchema
from src.utils import (
    ensure_dir,
    get_logger,
    init_logging,
    load_pickle,
    reduce_mem_usage,
    save_pickle,
    sha256_file,
    timer,
)


def test_read_yaml_rejects_non_mapping(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        _read_yaml(bad)


def test_read_yaml_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _read_yaml(tmp_path / "missing.yaml")


def test_load_config_with_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TRAINING_DEVICE", raising=False)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("{}", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.api.port == 8000
    assert cfg.training.n_folds == 5
    assert cfg.training.device == "auto"
    assert cfg.training.gpu_device_id == 0
    assert cfg.load_test.host.startswith("http://127.0.0.1:")
    load_config.cache_clear()


def test_load_config_rejects_invalid_training_device(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("TRAINING_DEVICE", raising=False)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("training:\n  device: invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training.device must be one of auto, cpu, cuda"):
        load_config(cfg_file)
    load_config.cache_clear()


def test_load_config_rejects_invalid_training_device_env(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("training:\n  device: auto\n", encoding="utf-8")
    monkeypatch.setenv("TRAINING_DEVICE", "invalid")
    with pytest.raises(ValueError, match="training.device must be one of auto, cpu, cuda"):
        load_config(cfg_file)
    load_config.cache_clear()
    monkeypatch.delenv("TRAINING_DEVICE", raising=False)


def test_load_config_training_device_env_override(tmp_path: Path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("training:\n  device: auto\n", encoding="utf-8")
    monkeypatch.setenv("TRAINING_DEVICE", "cpu")
    cfg = load_config(cfg_file)
    assert cfg.training.device == "cpu"
    load_config.cache_clear()
    monkeypatch.delenv("TRAINING_DEVICE", raising=False)


def test_load_config_custom_values(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        (
            "api:\n  port: 8123\n  workers: 2\n"
            "training:\n  n_folds: 3\n  enable_xgboost: true\n"
            "load_test:\n  host: http://127.0.0.1:8123\n  users: 10\n"
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.api.port == 8123
    assert cfg.api.workers == 2
    assert cfg.training.n_folds == 3
    assert cfg.training.enable_xgboost is True
    assert cfg.load_test.users == 10
    load_config.cache_clear()


def test_feature_schema_roles_and_nullable():
    df = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0], "A": [1.0], "B": [None]})
    schema = FeatureSchema.from_dataframe(df, version="v-test")
    roles = {c.name: c.role for c in schema.columns}
    nullable = {c.name: c.nullable for c in schema.columns}
    assert schema.version == "v-test"
    assert roles["SK_ID_CURR"] == "id"
    assert roles["TARGET"] == "target"
    assert roles["A"] == "feature"
    assert nullable["B"] is True


def test_utils_pickle_hash_logger_and_timer(tmp_path: Path):
    # ensure_dir + save/load pickle + hash
    target_dir = tmp_path / "nested" / "dir"
    ensure_dir(target_dir)
    pkl = target_dir / "obj.pkl"
    payload = {"x": 1, "y": [1, 2, 3]}
    save_pickle(payload, pkl)
    loaded = load_pickle(pkl)
    assert loaded == payload
    digest = sha256_file(pkl)
    assert isinstance(digest, str) and len(digest) == 64

    # init logging with file and no duplicate handlers for same file
    log_file = tmp_path / "logs" / "test.log"
    init_logging(log_file)
    base = logging.getLogger("home_credit")
    before = len(base.handlers)
    init_logging(log_file)
    after = len(base.handlers)
    assert after == before
    lg = get_logger("unit")
    lg.info("hello")

    # timer context runs without error
    with timer("unit-test"):
        pass


def test_reduce_mem_usage_downcasts_numeric_and_object():
    frame = pd.DataFrame(
        {
            "small_int": pd.Series([1, 2, 3], dtype="int64"),
            "small_float": pd.Series([1.5, 2.5, 3.5], dtype="float64"),
            "obj": ["a", "b", "a"],
        }
    )
    reduced = reduce_mem_usage(frame, verbose=False)
    assert str(reduced["small_int"].dtype) in {"int8", "int16", "int32"}
    assert str(reduced["small_float"].dtype).startswith("float")
    assert str(reduced["obj"].dtype) == "category"


def test_inference_record_to_feature_dict():
    rec = InferenceRecord(SK_ID_CURR=123, CUSTOM_COL=99)
    payload = rec.to_feature_dict()
    assert payload["SK_ID_CURR"] == 123
    assert payload["CUSTOM_COL"] == 99
