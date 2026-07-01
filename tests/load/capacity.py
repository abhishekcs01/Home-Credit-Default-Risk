from __future__ import annotations


def estimate_target_rps(max_concurrent_inferences: int, inference_seconds: float = 2.0) -> float:
    seconds = max(0.5, float(inference_seconds))
    return max(1.0, float(max_concurrent_inferences) / seconds)


def stagger_delay_seconds(
    users: int,
    target_rps: float,
    *,
    rng_uniform: float,
) -> float:
    """Spread each user's first request across the full capacity window."""
    active_users = max(1, int(users))
    rps = max(1.0, float(target_rps))
    spread = active_users / rps
    return max(0.0, float(rng_uniform) * spread)


def scale_wait_bounds(
    users: int,
    target_rps: float,
    *,
    min_floor: float = 5.0,
    inference_seconds: float = 2.0,
) -> tuple[float, float]:
    active_users = max(1, int(users))
    rps = max(1.0, float(target_rps))
    cycle = active_users / rps
    cycle = max(float(min_floor), cycle + float(inference_seconds))
    lo = max(float(min_floor), cycle * 0.45)
    hi = max(lo + 1.0, cycle * 0.55)
    return lo, hi
