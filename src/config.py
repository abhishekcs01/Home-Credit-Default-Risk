"""Thin compatibility layer: re-exports the repo-root ``config`` module.

Use ``from src import config`` in ``src/``, tests, and notebooks. All path and hyperparameter
constants are defined once in ``config.py`` at the project root; this file does not duplicate them.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("home_credit_project_config", _ROOT / "config.py")
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load project configuration at {_ROOT / 'config.py'}")
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("_")})
