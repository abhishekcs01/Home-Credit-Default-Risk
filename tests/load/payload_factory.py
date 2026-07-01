from __future__ import annotations

import copy
import json
from pathlib import Path
from random import Random
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_DIR = PROJECT_ROOT / "examples" / "payloads"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _jitter_numeric_fields(
    record: dict[str, Any], rng: Random, *, min_scale: float = 0.9, max_scale: float = 1.1
) -> None:
    for key, value in list(record.items()):
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if key == "SK_ID_CURR":
                continue
            scale = rng.uniform(min_scale, max_scale)
            record[key] = int(max(0, round(value * scale)))
        elif isinstance(value, float):
            scale = rng.uniform(min_scale, max_scale)
            record[key] = float(max(0.0, value * scale))


def _drop_non_numeric_features(record: dict[str, Any]) -> None:
    keys_to_remove = []
    for key, value in record.items():
        if key == "SK_ID_CURR":
            continue
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        record.pop(key, None)


def build_payload_pool(*, pool_size: int, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    rng = Random(seed)
    minimal = _load_json(PAYLOAD_DIR / "minimal_request.json")
    full = _load_json(PAYLOAD_DIR / "full_application_request.json")
    batch = _load_json(PAYLOAD_DIR / "batch_request.json")

    pools: dict[str, list[dict[str, Any]]] = {"minimal": [], "full": [], "batch": []}

    for idx in range(pool_size):
        p_min = copy.deepcopy(minimal)
        p_full = copy.deepcopy(full)
        p_batch = copy.deepcopy(batch)

        for rec in p_min.get("records", []):
            rec["SK_ID_CURR"] = int(rec.get("SK_ID_CURR", 100000)) + idx + rng.randint(0, 100)
            _jitter_numeric_fields(rec, rng)
            _drop_non_numeric_features(rec)
        for rec in p_full.get("records", []):
            rec["SK_ID_CURR"] = int(rec.get("SK_ID_CURR", 200000)) + idx + rng.randint(0, 100)
            _jitter_numeric_fields(rec, rng)
            _drop_non_numeric_features(rec)
        for rec in p_batch.get("records", []):
            rec["SK_ID_CURR"] = int(rec.get("SK_ID_CURR", 300000)) + idx * 10 + rng.randint(0, 10)
            _jitter_numeric_fields(rec, rng, min_scale=0.85, max_scale=1.15)
            _drop_non_numeric_features(rec)

        pools["minimal"].append(p_min)
        pools["full"].append(p_full)
        pools["batch"].append(p_batch)

    return pools
