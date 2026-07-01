from __future__ import annotations

from tests.load.capacity import estimate_target_rps, scale_wait_bounds, stagger_delay_seconds


def test_scale_wait_bounds_grows_with_users():
    lo, hi = scale_wait_bounds(20_000, estimate_target_rps(16, 2.0), min_floor=5.0, inference_seconds=2.0)
    assert lo > 1000.0
    assert hi > lo


def test_stagger_delay_spreads_first_requests():
    delay = stagger_delay_seconds(20_000, estimate_target_rps(16, 2.0), rng_uniform=0.5)
    assert 1000.0 < delay < 1500.0


def test_scale_wait_bounds_small_user_count():
    lo, hi = scale_wait_bounds(50, estimate_target_rps(16, 2.0), min_floor=5.0, inference_seconds=2.0)
    assert lo >= 5.0
    assert hi > lo
