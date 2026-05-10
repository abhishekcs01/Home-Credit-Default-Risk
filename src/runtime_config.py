"""API / Locust runtime settings from ``configs/config.yaml`` plus env overrides.

Canonical filesystem paths and training constants remain in the repo-root ``config.py``;
``src.config`` re-exports that module for ``import src.config``. Defaults such as
``model.bundle_path`` align with ``MODEL_BUNDLE_PATH`` unless overridden in YAML or env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src import config


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class APISettings:
    title: str
    version: str
    host: str
    port: int
    workers: int
    startup_load_model: bool
    max_request_body_bytes: int


@dataclass(frozen=True)
class ModelSettings:
    bundle_path: Path


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    structured: bool
    logger_name: str


@dataclass(frozen=True)
class LoadTestSettings:
    host: str
    users: int
    spawn_rate: int
    run_time: str
    min_wait_seconds: float
    max_wait_seconds: float
    locust_web_host: str
    locust_web_port: int


@dataclass(frozen=True)
class RuntimeSettings:
    api: APISettings
    model: ModelSettings
    logging: LoggingSettings
    load_test: LoadTestSettings


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing runtime config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be a mapping: {path}")
    return payload


@lru_cache(maxsize=1)
def load_runtime_settings(path: Path | None = None) -> RuntimeSettings:
    cfg_path = path or (config.PROJECT_ROOT / "configs" / "config.yaml")
    payload = _read_yaml(cfg_path)

    api = payload.get("api", {})
    model = payload.get("model", {})
    logging_cfg = payload.get("logging", {})
    load_test = payload.get("load_test", {})

    default_bundle = config.MODEL_BUNDLE_PATH
    configured_bundle = Path(model.get("bundle_path", str(default_bundle)))
    bundle_path = configured_bundle if configured_bundle.is_absolute() else (config.PROJECT_ROOT / configured_bundle)

    max_body_default = int(api.get("max_request_body_bytes", 6 * 1024 * 1024))
    max_body_env = os.getenv("HOME_CREDIT_API_MAX_REQUEST_BODY_BYTES")
    api_settings = APISettings(
        title=str(api.get("title", "Home Credit Default Risk API")),
        version=str(api.get("version", "1.0.0")),
        host=str(os.getenv("HOME_CREDIT_API_HOST", api.get("host", "0.0.0.0"))),
        port=int(os.getenv("HOME_CREDIT_API_PORT", api.get("port", 8000))),
        workers=int(os.getenv("HOME_CREDIT_API_WORKERS", api.get("workers", 1))),
        startup_load_model=_as_bool(
            os.getenv("HOME_CREDIT_API_STARTUP_LOAD_MODEL"),
            bool(api.get("startup_load_model", True)),
        ),
        max_request_body_bytes=int(max_body_env) if max_body_env is not None else max_body_default,
    )

    model_settings = ModelSettings(bundle_path=bundle_path)

    logging_settings = LoggingSettings(
        level=str(logging_cfg.get("level", "INFO")).upper(),
        structured=bool(logging_cfg.get("structured", True)),
        logger_name=str(logging_cfg.get("logger_name", "home_credit.api")),
    )

    # Locust/load tests call this URL; default tracks api.port so changing one port doesn’t desync the other.
    _default_load_host = f"http://127.0.0.1:{api_settings.port}"
    load_test_settings = LoadTestSettings(
        host=str(os.getenv("HOME_CREDIT_LOAD_HOST", load_test.get("host", _default_load_host))),
        users=int(os.getenv("HOME_CREDIT_LOAD_USERS", load_test.get("users", 25))),
        spawn_rate=int(os.getenv("HOME_CREDIT_LOAD_SPAWN_RATE", load_test.get("spawn_rate", 5))),
        run_time=str(os.getenv("HOME_CREDIT_LOAD_RUN_TIME", load_test.get("run_time", "2m"))),
        min_wait_seconds=float(
            os.getenv("HOME_CREDIT_LOAD_MIN_WAIT_SECONDS", load_test.get("min_wait_seconds", 0.2))
        ),
        max_wait_seconds=float(
            os.getenv("HOME_CREDIT_LOAD_MAX_WAIT_SECONDS", load_test.get("max_wait_seconds", 1.0))
        ),
        locust_web_host=str(os.getenv("HOME_CREDIT_LOCUST_WEB_HOST", load_test.get("locust_web_host", "127.0.0.1"))),
        locust_web_port=int(os.getenv("HOME_CREDIT_LOCUST_WEB_PORT", load_test.get("locust_web_port", 8089))),
    )

    return RuntimeSettings(
        api=api_settings,
        model=model_settings,
        logging=logging_settings,
        load_test=load_test_settings,
    )
