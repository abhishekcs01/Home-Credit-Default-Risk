from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ApiSettings:
    host: str
    port: int
    workers: int
    startup_load_model: bool
    version: str
    title: str
    inference_batch_size: int
    max_request_body_bytes: int
    inference_timeout_seconds: float
    prediction_retry_count: int
    max_concurrent_inferences: int
    max_pending_requests: int
    queue_wait_seconds: float
    pending_acquire_seconds: float
    max_inflight_requests: int
    limit_concurrency: int
    backlog: int
    max_batch_size: int = 2048


@dataclass(frozen=True)
class TrainingSettings:
    random_state: int
    device: str
    gpu_device_id: int
    n_folds: int
    ohe_max_categories: int
    miss_drop_threshold: float
    prune_bottom_frac: float
    blend_weight_lgb: float
    blend_weight_cat: float
    blend_weight_xgb: float
    enable_xgboost: bool
    enable_stacking: bool
    enable_calibration: bool
    enable_optuna: bool
    enable_shap: bool


@dataclass(frozen=True)
class LoadTestSettings:
    host: str
    users: int
    spawn_rate: int
    run_time: str
    min_wait_seconds: float
    max_wait_seconds: float
    expected_inference_seconds: float
    auto_scale_wait: bool
    stagger_first_request: bool
    retry_count: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class AppConfig:
    api: ApiSettings
    training: TrainingSettings
    load_test: LoadTestSettings


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or (PROJECT_ROOT / "configs" / "config.yaml")
    payload = _read_yaml(cfg_path)

    api_cfg = payload.get("api", {})
    tr_cfg = payload.get("training", {})
    lt_cfg = payload.get("load_test", {})

    api = ApiSettings(
        host=str(api_cfg.get("host", "0.0.0.0")),
        port=int(api_cfg.get("port", 8000)),
        workers=int(api_cfg.get("workers", 1)),
        startup_load_model=bool(api_cfg.get("startup_load_model", True)),
        version=str(api_cfg.get("version", "1.0.0")),
        title=str(api_cfg.get("title", "Home Credit Default Risk API")),
        inference_batch_size=int(api_cfg.get("inference_batch_size", 512)),
        max_request_body_bytes=int(api_cfg.get("max_request_body_bytes", 6 * 1024 * 1024)),
        inference_timeout_seconds=float(api_cfg.get("inference_timeout_seconds", 25.0)),
        prediction_retry_count=int(api_cfg.get("prediction_retry_count", 1)),
        max_concurrent_inferences=int(
            api_cfg.get("max_concurrent_inferences", max(4, (os.cpu_count() or 4)))
        ),
        max_pending_requests=int(api_cfg.get("max_pending_requests", 256)),
        queue_wait_seconds=float(api_cfg.get("queue_wait_seconds", 120.0)),
        pending_acquire_seconds=float(api_cfg.get("pending_acquire_seconds", 2.0)),
        max_inflight_requests=int(api_cfg.get("max_inflight_requests", 512)),
        limit_concurrency=int(api_cfg.get("limit_concurrency", 256)),
        backlog=int(api_cfg.get("backlog", 2048)),
        max_batch_size=int(api_cfg.get("max_batch_size", 2048)),
    )

    device = str(os.environ.get("TRAINING_DEVICE", tr_cfg.get("device", "auto"))).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"training.device must be one of auto, cpu, cuda; got {device!r}")

    training = TrainingSettings(
        random_state=int(tr_cfg.get("random_state", 42)),
        device=device,
        gpu_device_id=int(tr_cfg.get("gpu_device_id", 0)),
        n_folds=int(tr_cfg.get("n_folds", 5)),
        ohe_max_categories=int(tr_cfg.get("ohe_max_categories", 15)),
        miss_drop_threshold=float(tr_cfg.get("miss_drop_threshold", 0.70)),
        prune_bottom_frac=float(tr_cfg.get("prune_bottom_frac", 0.06)),
        blend_weight_lgb=float(tr_cfg.get("blend_weight_lgb", 0.5)),
        blend_weight_cat=float(tr_cfg.get("blend_weight_cat", 0.5)),
        blend_weight_xgb=float(tr_cfg.get("blend_weight_xgb", 0.0)),
        enable_xgboost=bool(tr_cfg.get("enable_xgboost", False)),
        enable_stacking=bool(tr_cfg.get("enable_stacking", True)),
        enable_calibration=bool(tr_cfg.get("enable_calibration", True)),
        enable_optuna=bool(tr_cfg.get("enable_optuna", False)),
        enable_shap=bool(tr_cfg.get("enable_shap", False)),
    )

    load_test = LoadTestSettings(
        host=str(lt_cfg.get("host", f"http://127.0.0.1:{api.port}")),
        users=int(lt_cfg.get("users", 25)),
        spawn_rate=int(lt_cfg.get("spawn_rate", 5)),
        run_time=str(lt_cfg.get("run_time", "2m")),
        min_wait_seconds=float(lt_cfg.get("min_wait_seconds", 0.2)),
        max_wait_seconds=float(lt_cfg.get("max_wait_seconds", 1.0)),
        expected_inference_seconds=float(lt_cfg.get("expected_inference_seconds", 2.0)),
        auto_scale_wait=bool(lt_cfg.get("auto_scale_wait", True)),
        stagger_first_request=bool(lt_cfg.get("stagger_first_request", True)),
        retry_count=int(lt_cfg.get("retry_count", 6)),
        retry_backoff_seconds=float(lt_cfg.get("retry_backoff_seconds", 5.0)),
    )

    return AppConfig(api=api, training=training, load_test=load_test)
