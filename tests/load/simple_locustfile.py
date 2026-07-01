from __future__ import annotations

import random
from typing import Any

import gevent
from locust import events, task
from locust.contrib.fasthttp import FastHttpUser

from src.config import load_config
from tests.load.capacity import estimate_target_rps, scale_wait_bounds, stagger_delay_seconds
from tests.load.payload_factory import build_payload_pool

cfg = load_config()
pool = build_payload_pool(pool_size=200, seed=42)


class HomeCreditInferenceUser(FastHttpUser):
    host = cfg.load_test.host
    wait_min = cfg.load_test.min_wait_seconds
    wait_max = cfg.load_test.max_wait_seconds
    target_users = cfg.load_test.users
    target_rps = estimate_target_rps(
        cfg.api.max_concurrent_inferences,
        cfg.load_test.expected_inference_seconds,
    )
    stagger_first_request = cfg.load_test.stagger_first_request
    retry_count = cfg.load_test.retry_count
    retry_backoff_seconds = cfg.load_test.retry_backoff_seconds
    connection_timeout = 60.0
    network_timeout = 600.0

    def on_start(self) -> None:
        if not self.stagger_first_request:
            return
        rng = random.Random(id(self))
        delay = stagger_delay_seconds(
            self.target_users,
            self.target_rps,
            rng_uniform=rng.random(),
        )
        if delay > 0:
            gevent.sleep(delay)

    def wait_time(self) -> float:
        return random.uniform(self.wait_min, self.wait_max)

    def _pick_payload(self, kind: str) -> dict[str, Any]:
        return random.choice(pool[kind])

    def _post_predict(self, kind: str, name: str) -> None:
        payload = self._pick_payload(kind)
        attempts = max(0, int(self.retry_count)) + 1
        for attempt in range(attempts):
            with self.client.post(
                "/predict",
                json=payload,
                name=name,
                catch_response=True,
            ) as response:
                if response.status_code == 0:
                    if attempt + 1 < attempts:
                        gevent.sleep(self.retry_backoff_seconds * (attempt + 1))
                        continue
                    response.failure("connection error")
                    return
                if response.status_code == 503 and attempt + 1 < attempts:
                    gevent.sleep(self.retry_backoff_seconds * (attempt + 1))
                    continue
                if response.status_code >= 500:
                    response.failure(f"server error {response.status_code}")
                    return
                if response.status_code != 200:
                    response.failure(f"unexpected status {response.status_code}")
                    return
                response.success()
                return

    @task(65)
    def predict_minimal(self):
        self._post_predict("minimal", "predict_minimal")

    @task(25)
    def predict_full(self):
        self._post_predict("full", "predict_full")

    @task(10)
    def predict_batch(self):
        self._post_predict("batch", "predict_batch")


@events.test_start.add_listener
def _scale_wait_for_user_count(environment, **kwargs) -> None:
    runner = environment.runner
    users = getattr(runner, "target_user_count", None) or cfg.load_test.users
    spawn_rate = float(getattr(runner, "spawn_rate", None) or cfg.load_test.spawn_rate or 1)
    target_rps = estimate_target_rps(
        cfg.api.max_concurrent_inferences,
        cfg.load_test.expected_inference_seconds,
    )

    HomeCreditInferenceUser.target_users = users
    HomeCreditInferenceUser.target_rps = target_rps
    HomeCreditInferenceUser.stagger_first_request = cfg.load_test.stagger_first_request
    HomeCreditInferenceUser.retry_count = cfg.load_test.retry_count
    HomeCreditInferenceUser.retry_backoff_seconds = cfg.load_test.retry_backoff_seconds

    if cfg.load_test.auto_scale_wait:
        wait_min, wait_max = scale_wait_bounds(
            users,
            target_rps,
            min_floor=cfg.load_test.min_wait_seconds,
            inference_seconds=cfg.load_test.expected_inference_seconds,
        )
        HomeCreditInferenceUser.wait_min = wait_min
        HomeCreditInferenceUser.wait_max = wait_max
        print(
            f"[load-test] Scaled think time for {users} users: "
            f"{wait_min:.1f}-{wait_max:.1f}s (capacity ~{target_rps:.1f} RPS)"
        )

    spread_s = users / target_rps
    print(
        f"[load-test] First-request stagger window: 0-{spread_s:.0f}s "
        f"(spawn_rate={spawn_rate:.1f}/s, retries={cfg.load_test.retry_count})"
    )
    if spawn_rate > target_rps:
        print(
            f"[load-test] Warning: spawn_rate ({spawn_rate:.1f}) exceeds capacity "
            f"({target_rps:.1f} RPS). Stagger + retries will absorb the ramp burst."
        )
