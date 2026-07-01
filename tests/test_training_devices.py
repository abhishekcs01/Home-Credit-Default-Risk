from __future__ import annotations

import logging
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.models.train as mt
from src.models.train import TrainingDevices, log_training_devices, resolve_training_devices


def _cpu_devices(requested: str = "cpu") -> TrainingDevices:
    return TrainingDevices("cpu", "CPU", None, "cpu", 0, requested)


def _gpu_devices(requested: str = "cuda") -> TrainingDevices:
    return TrainingDevices("cuda", "GPU", "0", "cuda", 0, requested)


def test_resolve_cpu_mode():
    devices = resolve_training_devices("cpu", gpu_device_id=0)
    assert devices.lgb_device == "cpu"
    assert devices.catboost_task_type == "CPU"
    assert devices.xgb_device == "cpu"


def test_resolve_cuda_mode(monkeypatch):
    monkeypatch.setattr(mt, "_lightgbm_cuda_available", lambda: True)
    devices = resolve_training_devices("cuda", gpu_device_id=0)
    assert devices.lgb_device == "cuda"
    assert devices.catboost_task_type == "GPU"
    assert devices.catboost_devices == "0"
    assert devices.xgb_device == "cuda"


def test_resolve_auto_uses_gpu_when_available(monkeypatch):
    monkeypatch.setattr(mt, "_nvidia_gpu_available", lambda: True)
    monkeypatch.setattr(mt, "_lightgbm_cuda_available", lambda: True)
    devices = resolve_training_devices("auto", gpu_device_id=0)
    assert devices.lgb_device == "cuda"
    assert devices.catboost_task_type == "GPU"


def test_resolve_auto_falls_back_lgb_to_cpu_without_cuda_build(monkeypatch):
    monkeypatch.setattr(mt, "_nvidia_gpu_available", lambda: True)
    monkeypatch.setattr(mt, "_lightgbm_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="LightGBM CUDA is required"):
        resolve_training_devices("auto", gpu_device_id=0)


def test_resolve_auto_falls_back_to_cpu_without_gpu(monkeypatch):
    monkeypatch.setattr(mt, "_nvidia_gpu_available", lambda: False)
    devices = resolve_training_devices("auto", gpu_device_id=0)
    assert devices.lgb_device == "cpu"
    assert devices.catboost_task_type == "CPU"


def test_invalid_device_raises():
    with pytest.raises(ValueError, match="training.device must be one of auto, cpu, cuda"):
        resolve_training_devices("tpu", gpu_device_id=0)


def test_param_builders_map_gpu_settings():
    devices = _gpu_devices()
    lgb_params = mt._lgb_binary_params(devices, 1.0, random_state=42)
    cat_params = mt._catboost_binary_params(devices, 1.0, random_state=42)
    xgb_params = mt._xgb_binary_params(devices, 1.0, random_state=42)

    assert lgb_params["device"] == "cuda"
    assert lgb_params["gpu_device_id"] == 0
    assert "force_col_wise" not in lgb_params
    assert cat_params["task_type"] == "GPU"
    assert cat_params["devices"] == "0"
    assert xgb_params["device"] == "cuda"
    assert xgb_params["tree_method"] == "hist"


def test_param_builders_map_cpu_settings():
    devices = _cpu_devices()
    lgb_params = mt._lgb_binary_params(devices, 1.0, random_state=42)
    cat_params = mt._catboost_binary_params(devices, 1.0, random_state=42)
    xgb_params = mt._xgb_binary_params(devices, 1.0, random_state=42)

    assert lgb_params["device"] == "cpu"
    assert lgb_params["force_col_wise"] is True
    assert cat_params["task_type"] == "CPU"
    assert "devices" not in cat_params
    assert xgb_params["device"] == "cpu"


def test_log_training_devices_emits_backend_lines(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test.training.devices")
    log_training_devices(logger, _gpu_devices())
    assert "LIGHTGBM BACKEND: CUDA" in caplog.text
    assert "CATBOOST BACKEND: GPU" in caplog.text
    assert "XGBOOST BACKEND: CUDA" in caplog.text


def test_log_training_devices_warns_when_gpu_unavailable(caplog):
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("test.training.devices.warn")
    devices = TrainingDevices("cpu", "CPU", None, "cpu", 0, "auto")
    log_training_devices(logger, devices)
    assert "GPU requested but no NVIDIA GPU detected" in caplog.text


def test_resolve_cuda_requires_lgb_cuda_build(monkeypatch):
    monkeypatch.setattr(mt, "_lightgbm_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="LightGBM CUDA is required"):
        resolve_training_devices("cuda", gpu_device_id=0)


def test_nvidia_gpu_available_true(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "GPU Name\n"

    monkeypatch.setattr(mt.subprocess, "run", lambda *args, **kwargs: _Result())
    assert mt._nvidia_gpu_available() is True


def test_nvidia_gpu_available_false_on_error(monkeypatch):
    monkeypatch.setattr(mt.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    assert mt._nvidia_gpu_available() is False


def test_nvidia_gpu_available_false_on_empty_stdout(monkeypatch):
    class _Result:
        returncode = 0
        stdout = "\n"

    monkeypatch.setattr(mt.subprocess, "run", lambda *args, **kwargs: _Result())
    assert mt._nvidia_gpu_available() is False


def test_nvidia_gpu_available_false_on_nonzero_exit(monkeypatch):
    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(mt.subprocess, "run", lambda *args, **kwargs: _Result())
    assert mt._nvidia_gpu_available() is False


def test_resolve_cuda_uses_configured_gpu_id(monkeypatch):
    monkeypatch.setattr(mt, "_lightgbm_cuda_available", lambda: True)
    devices = resolve_training_devices("cuda", gpu_device_id=2)
    assert devices.gpu_device_id == 2
    assert devices.catboost_devices == "2"
    assert devices.lgb_device == "cuda"


def test_lightgbm_cuda_available_false_without_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(mt, "_nvidia_gpu_available", lambda: False)
    mt._lightgbm_cuda_available.cache_clear()
    assert mt._lightgbm_cuda_available() is False


def test_lightgbm_cuda_available_false_on_lgbm_error(monkeypatch):
    monkeypatch.setattr(mt, "_nvidia_gpu_available", lambda: True)

    def _raise(*_args, **_kwargs):
        raise mt.lgb.basic.LightGBMError("CUDA Tree Learner was not enabled in this build.")

    monkeypatch.setattr(mt.lgb, "train", _raise)
    mt._lightgbm_cuda_available.cache_clear()
    assert mt._lightgbm_cuda_available() is False


def test_training_devices_from_config(monkeypatch):
    monkeypatch.setattr(
        mt,
        "load_config",
        lambda: types.SimpleNamespace(training=types.SimpleNamespace(device="cpu", gpu_device_id=0)),
    )
    devices = mt._training_devices_from_config()
    assert devices.lgb_device == "cpu"


def test_optuna_tune_lgb_returns_none_when_optuna_missing(monkeypatch):
    monkeypatch.setattr(mt, "optuna", None)
    x = np.random.rand(20, 4)
    y = np.array([0, 1] * 10, dtype=np.int32)
    assert mt._optuna_tune_lgb(x, y, x, y, random_state=42) is None


def test_nvidia_gpu_available_false_on_timeout(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        mt.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("nvidia-smi", 5)),
    )
    assert mt._nvidia_gpu_available() is False


def test_train_main_invokes_run_training(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(mt, "init_logging", lambda: None)
    monkeypatch.setattr(mt, "run_training", lambda: called.append("ok") or Path("model.pkl"))
    mt.main()
    assert called == ["ok"]


def test_nvidia_gpu_available_false_on_os_error(monkeypatch):
    monkeypatch.setattr(mt.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unavailable")))
    assert mt._nvidia_gpu_available() is False


def test_evaluate_models_accepts_explicit_devices():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(80),
            "TARGET": rng.integers(0, 2, size=80),
            "F1": rng.normal(size=80),
            "CAT": rng.choice(["A", "B"], size=80),
        }
    )
    result = mt.evaluate_models(df, devices=_cpu_devices(), ohe_max_categories=4, prune_bottom_frac=0.0)
    assert 0.0 <= result["auc_lgb"] <= 1.0


def test_train_kfold_accepts_explicit_devices(monkeypatch):
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(60),
            "TARGET": rng.integers(0, 2, size=60),
            "F1": rng.normal(size=60),
            "CAT": rng.choice(["A", "B"], size=60),
        }
    )
    monkeypatch.setattr(mt, "CatBoostClassifier", None)
    monkeypatch.setattr(mt, "xgb", None)
    monkeypatch.setattr(
        mt,
        "_train_lgb",
        lambda x_tr, y_tr, x_va, y_va, random_state, **kwargs: types.SimpleNamespace(
            best_iteration=1,
            predict=lambda x, num_iteration=None: np.full(len(x), 0.5),
            feature_importance=lambda importance_type="gain": np.ones(x_tr.shape[1]),
        ),
    )
    out = mt.train_kfold_lightgbm_ensemble(
        df,
        n_splits=2,
        random_state=3,
        ohe_max_categories=4,
        devices=_cpu_devices(),
        blend_weights=(1.0,),
        feature_keep_indices=np.array([0, 1], dtype=int),
        enable_stacking=False,
        enable_calibration=False,
        enable_optuna=False,
        enable_shap=False,
    )
    assert len(out["fold_models"]) == 2


def test_run_training_resolves_devices_before_data_check(monkeypatch):
    devices = _cpu_devices()
    recorded: list[object] = []
    missing_train = Path("/does/not/exist/merged_train.pkl")
    monkeypatch.setattr(mt, "resolve_training_devices", lambda device, gpu_device_id: devices)
    monkeypatch.setattr(mt, "log_training_devices", lambda logger, value: recorded.append(value))
    monkeypatch.setattr(mt.config, "MERGED_TRAIN_PATH", missing_train)
    monkeypatch.setattr(
        "src.data.pipeline.ensure_processed_datasets",
        lambda **kwargs: (missing_train, Path("/does/not/exist/merged_test.pkl")),
    )
    monkeypatch.setattr(
        mt,
        "load_config",
        lambda: types.SimpleNamespace(
            training=types.SimpleNamespace(device="cpu", gpu_device_id=0),
            api=types.SimpleNamespace(version="test"),
        ),
    )

    with pytest.raises(FileNotFoundError):
        mt.run_training()

    assert recorded == [devices]


def test_run_training_logs_training_started(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="home_credit.train_model")
    devices = _cpu_devices()
    train_pkl = tmp_path / "merged_train.pkl"
    pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0], "F_0": [1.0]}).to_pickle(train_pkl)

    monkeypatch.setattr(mt, "resolve_training_devices", lambda device, gpu_device_id: devices)
    monkeypatch.setattr(mt, "log_training_devices", lambda logger, value: None)
    monkeypatch.setattr(mt.config, "MERGED_TRAIN_PATH", train_pkl)
    monkeypatch.setattr(
        mt,
        "load_config",
        lambda: types.SimpleNamespace(
            training=types.SimpleNamespace(
                device="cpu",
                gpu_device_id=0,
                random_state=42,
                ohe_max_categories=15,
                miss_drop_threshold=0.7,
                prune_bottom_frac=0.06,
                n_folds=5,
                blend_weight_lgb=0.5,
                blend_weight_cat=0.5,
                blend_weight_xgb=0.0,
                enable_xgboost=False,
                enable_stacking=True,
                enable_calibration=True,
                enable_optuna=False,
                enable_shap=False,
            ),
            api=types.SimpleNamespace(version="test"),
        ),
    )

    def _stop_after_start(*_args, **_kwargs):
        raise RuntimeError("stop-after-start")

    monkeypatch.setattr(mt, "train_kfold_lightgbm_ensemble", _stop_after_start)

    with pytest.raises(RuntimeError, match="stop-after-start"):
        mt.run_training()

    assert "Training started" in caplog.text
