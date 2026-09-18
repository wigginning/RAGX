"""Circuit breaker (08-llm.md §8.4).

Granularity is per ``(provider, model)``. The design's §8.4.2 deque is a
simplified sketch; §8.4.1 mandates the production form: a **time-bucket**
window (one bucket per second, keeping ``window_s`` buckets) so only samples
inside the window count toward the failure rate.

State machine: CLOSED -> OPEN (failure_rate >= threshold AND samples >=
min_samples) -> HALF_OPEN (after ``open_s``) -> CLOSED on a successful probe,
or back to OPEN on a failed probe.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from ragx.core.settings import CircuitConfig


class CircuitBreaker:
    """Per-key circuit state with a sliding time-bucket failure window."""

    def __init__(self, cfg: CircuitConfig) -> None:
        self.cfg = cfg
        #: key -> "closed" | "open" | "half_open"
        self._state: dict[tuple[str, str], str] = {}
        #: key -> deque of (bucket_second, success: bool)
        self._buckets: dict[tuple[str, str], deque[tuple[int, bool]]] = defaultdict(
            lambda: deque()
        )
        self._opened_at: dict[tuple[str, str], float] = {}

    # -- state --------------------------------------------------------------
    def is_open(self, key: tuple[str, str]) -> bool:
        """True -> fast-fail (raise CircuitOpenError). False -> allow the call.

        A key in ``open`` for longer than ``open_s`` transitions to ``half_open``
        and is allowed through as a single probe.
        """
        state = self._state.get(key, "closed")
        if state == "open":
            if time.time() - self._opened_at[key] >= self.cfg.open_s:
                self._state[key] = "half_open"
                return False
            return True
        return False

    def record_success(self, key: tuple[str, str]) -> None:
        if self._state.get(key) == "half_open":
            self._state[key] = "closed"  # probe succeeded -> recovered
        self._append(key, True)

    def record_failure(self, key: tuple[str, str]) -> None:
        self._append(key, False)
        self._maybe_open(key)

    # -- internals ----------------------------------------------------------
    def _append(self, key: tuple[str, str], success: bool) -> None:
        now = int(time.time())
        self._buckets[key].append((now, success))
        self._prune(key, now)

    def _prune(self, key: tuple[str, str], now: int) -> None:
        """Drop samples older than the window (time-bucket semantics)."""
        cutoff = now - int(self.cfg.window_s)
        bucket = self._buckets[key]
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()

    def _maybe_open(self, key: tuple[str, str]) -> None:
        now = int(time.time())
        self._prune(key, now)
        window = self._buckets[key]
        if len(window) < self.cfg.min_samples:
            return
        failures = sum(1 for _, ok in window if not ok)
        if failures / len(window) >= self.cfg.failure_rate:
            self._state[key] = "open"
            self._opened_at[key] = time.time()

    # -- introspection -------------------------------------------------------
    def state(self, key: tuple[str, str]) -> str:
        return self._state.get(key, "closed")

    def snapshot(self) -> dict[str, Any]:
        """Observability: current state per key (08-llm.md §8.8)."""
        return {f"{p}/{m}": s for (p, m), s in self._state.items()}
