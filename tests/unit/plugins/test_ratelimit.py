"""RateLimiter unit tests (09-api.md §9.2.4, Agent D RX-RET-06).

The in-process backend needs no Redis service, so the token-bucket algorithm is
fully tested here. The Redis backend is tested for graceful **fail-open** when
Redis is unreachable (the API must degrade rather than 5xx).
"""

from __future__ import annotations

import asyncio

import pytest

from ragx.api.middleware.ratelimit import (
    InProcessRateLimiter,
    RateLimiter,
    RedisRateLimiter,
)
from ragx.core.exceptions import RateLimitError
from ragx.core.settings import SecurityConfig


def _cfg(*, rps: float = 10.0, burst: int = 5) -> SecurityConfig:
    return SecurityConfig(rate_limit_rps=rps, rate_limit_burst=burst)


async def test_allows_within_burst() -> None:
    """Up to `burst` tokens are available immediately."""
    rl = InProcessRateLimiter(_cfg(rps=1.0, burst=5))
    for _ in range(5):
        assert await rl.acquire("key") is None
    # 6th exceeds the burst -> rate limited with a positive retry-after
    wait = await rl.acquire("key")
    assert wait is not None and wait > 0


async def test_per_key_isolation() -> None:
    """Each key has its own bucket — exhausting one key does not affect another."""
    rl = InProcessRateLimiter(_cfg(rps=100.0, burst=1))
    assert await rl.acquire("a") is None
    assert await rl.acquire("a") is not None
    assert await rl.acquire("b") is None  # fresh bucket unaffected


async def test_refills_over_time() -> None:
    """Tokens refill at `rate` per second after the burst is exhausted."""
    rl = InProcessRateLimiter(_cfg(rps=10.0, burst=1))
    assert await rl.acquire("k") is None
    await asyncio.sleep(0.15)  # refill ~1.5 tokens
    assert await rl.acquire("k") is None  # enough refilled


async def test_add_tokens_consumes_bucket() -> None:
    """Token dimension: LLM-call token accounting eats into the bucket."""
    rl = InProcessRateLimiter(_cfg(rps=100.0, burst=10))
    await rl.add_tokens("k", 9)  # consume 9 of 10
    assert await rl.acquire("k") is None  # 1 left
    wait = await rl.acquire("k")
    assert wait is not None  # empty -> rate limited


async def test_check_or_raise_raises_1004() -> None:
    """Acquiring past the limit raises RateLimitError(1004)."""
    rl = RateLimiter.create(_cfg(rps=1.0, burst=1))
    await rl.check_or_raise("k")
    with pytest.raises(RateLimitError) as ei:
        await rl.check_or_raise("k")
    assert ei.value.code == 1004
    assert "retry_after" in ei.value.details


async def test_redis_backend_fails_open_on_unreachable() -> None:
    """Redis down -> acquire returns None (allowed) rather than raising.

    Use a closed port so the client cannot connect; the limiter must degrade
    to allow the request (09-api.md §9.2.4 fail-open rationale).
    """
    cfg = _cfg(rps=10.0, burst=5)
    rl = RedisRateLimiter(cfg, redis_url="redis://127.0.0.1:1/0")  # port 1 -> refused
    # no startup/ping — first acquire attempts the connection
    assert await rl.acquire("k") is None
    await rl.shutdown()
