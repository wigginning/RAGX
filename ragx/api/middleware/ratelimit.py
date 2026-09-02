"""Per-key token-bucket rate limiter (09-api.md §9.2.4).

Two backends:

* :class:`InProcessRateLimiter` — lite profile, process-internal; buckets live
  in a dict keyed by API key. Zero external dependencies.
* :class:`RedisRateLimiter` — full profile, cross-instance shared state; uses a
  Lua script for atomic check-and-consume so concurrent workers cannot race.

Both expose the same interface::

    rl = RateLimiter.create(config)
    wait = await rl.acquire(key, tokens=1)
    if wait is not None:
        raise RateLimitError(code=1004, ..., details={"retry_after": wait})

``acquire`` returns ``None`` (allowed) or the ``Retry-After`` seconds (rate
limited). The two dimensions mandated by §9.2.4 are covered:

* **request dimension** — ``acquire(key, tokens=1)`` for QPS
* **token dimension** — ``add_tokens(key, count)`` for LLM-call token accounting,
  back-filled by the Resilient Router after each LLM call
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ragx.api.errors import error_response
from ragx.core.exceptions import RAGXError, RateLimitError
from ragx.core.settings import SecurityConfig

logger = logging.getLogger("ragx.api.ratelimit")

# Lua script: atomic token-bucket check-and-consume.
# Returns 0 if allowed, or the wait time in seconds if rate-limited.
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local tokens = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local current = tonumber(data[1])
if current == nil then current = capacity end
local last_ts = tonumber(data[2])
if last_ts == nil then last_ts = now end

local elapsed = math.max(0, now - last_ts)
current = math.min(capacity, current + elapsed * rate)

if current >= tokens then
    current = current - tokens
    redis.call('HMSET', key, 'tokens', current, 'ts', now)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)
    return 0
else
    local needed = tokens - current
    local wait = needed / rate
    redis.call('HMSET', key, 'tokens', current, 'ts', now)
    redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)
    return tostring(wait)
end
"""

# Lua script: add tokens to a bucket (token dimension back-fill).
_ADD_TOKENS_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local add = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local current = tonumber(data[1])
if current == nil then current = capacity end
local last_ts = tonumber(data[2])
if last_ts == nil then last_ts = now end

local elapsed = math.max(0, now - last_ts)
current = math.min(capacity, current + elapsed * rate)
-- subtract the consumed tokens (token dimension: LLM calls eat into the bucket)
current = math.max(0, current - add)
redis.call('HMSET', key, 'tokens', current, 'ts', now)
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 1)
return tostring(current)
"""


class _TokenBucket:
    """In-process token bucket for one key."""

    __slots__ = ("capacity", "rate", "tokens", "last_refill", "_lock")

    def __init__(self, rate: float, capacity: float) -> None:
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    async def acquire(self, tokens: float = 1.0) -> float | None:
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return None
            deficit = tokens - self.tokens
            return deficit / self.rate

    async def add_tokens(self, count: float) -> None:
        async with self._lock:
            self._refill()
            self.tokens = max(0.0, self.tokens - count)


class InProcessRateLimiter:
    """Lite profile: process-internal token buckets."""

    def __init__(self, config: SecurityConfig) -> None:
        self.rate = config.rate_limit_rps
        self.burst = config.rate_limit_burst
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def _bucket(self, key: str) -> _TokenBucket:
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _TokenBucket(self.rate, self.burst)
            return self._buckets[key]

    async def acquire(self, key: str, tokens: float = 1.0) -> float | None:
        bucket = await self._bucket(key)
        return await bucket.acquire(tokens)

    async def add_tokens(self, key: str, count: float) -> None:
        bucket = await self._bucket(key)
        await bucket.add_tokens(count)


class RedisRateLimiter:
    """Full profile: Redis-backed token buckets (cross-instance shared)."""

    def __init__(self, config: SecurityConfig, redis_url: str = "") -> None:
        self.rate = config.rate_limit_rps
        self.burst = config.rate_limit_burst
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self._client: Any = None
        self._bucket_sha: str | None = None
        self._add_sha: str | None = None
        self._prefix = "ragx:rl"

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise RateLimitError(
                "redis package not installed (pip install ragx[redis])",
                code=1004, details={"error": str(exc)},
            ) from exc
        self._client = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._client

    async def _sha(self, script: str, attr: str) -> str:
        client = self._ensure_client()
        sha = getattr(self, attr)
        if sha is None:
            sha = await client.script_load(script)
            setattr(self, attr, sha)
        return sha

    async def acquire(self, key: str, tokens: float = 1.0) -> float | None:
        client = self._ensure_client()
        bucket_key = f"{self._prefix}:{key}"
        try:
            sha = await self._sha(_TOKEN_BUCKET_LUA, "_bucket_sha")
            result = await client.evalsha(
                sha, 1, bucket_key,
                self.rate, self.burst, float(tokens), time.time(),
            )
            if result == 0 or result == "0":
                return None
            return float(result)
        except Exception as exc:
            # Redis down: fail open (allow the request) and log a warning.
            logger.warning("RedisRateLimiter.acquire: Redis unavailable, "
                           "failing open: %s", exc)
            return None

    async def add_tokens(self, key: str, count: float) -> None:
        client = self._ensure_client()
        bucket_key = f"{self._prefix}:{key}"
        try:
            sha = await self._sha(_ADD_TOKENS_LUA, "_add_sha")
            await client.evalsha(
                sha, 1, bucket_key,
                self.rate, self.burst, float(count), time.time(),
            )
        except Exception as exc:
            logger.warning("RedisRateLimiter.add_tokens: Redis unavailable: %s", exc)

    async def shutdown(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None


class RateLimiter:
    """Factory: picks the backend based on the Redis URL availability."""

    def __init__(self, backend: InProcessRateLimiter | RedisRateLimiter) -> None:
        self._backend = backend

    @classmethod
    def create(
        cls, config: SecurityConfig, *, redis_url: str = "",
    ) -> RateLimiter:
        """Create the appropriate backend (lite=in-process, full=Redis)."""
        if redis_url:
            return cls(RedisRateLimiter(config, redis_url=redis_url))
        return cls(InProcessRateLimiter(config))

    async def acquire(self, key: str, tokens: float = 1.0) -> float | None:
        """Return ``None`` (allowed) or ``Retry-After`` seconds (rate limited)."""
        return await self._backend.acquire(key, tokens)

    async def add_tokens(self, key: str, count: float) -> None:
        """Back-fill token consumption from LLM calls (token dimension)."""
        await self._backend.add_tokens(key, count)

    async def check_or_raise(self, key: str, tokens: float = 1.0) -> None:
        """Acquire tokens or raise ``RateLimitError(1004)``."""
        wait = await self.acquire(key, tokens)
        if wait is not None:
            raise RateLimitError(
                code=1004,
                message="rate limit exceeded",
                details={
                    "key": key,
                    "retry_after": round(wait, 3),
                    "tokens_requested": tokens,
                },
            )

    async def shutdown(self) -> None:
        if hasattr(self._backend, "shutdown"):
            await self._backend.shutdown()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-key token-bucket enforcement at the request dimension (§9.2.4).

    Relies on ``AuthMiddleware`` having run first: the request's auth context
    provides the bucket key (``key_id`` for authenticated callers, the literal
    ``"anonymous"`` otherwise). A rate-limited request raises
    ``RateLimitError(1004)``, mapped to HTTP 429 with ``Retry-After`` by the
    unified error handler.
    """

    def __init__(self, app: Any, *, limiter: RateLimiter) -> None:
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            auth = getattr(request.state, "auth", None)
            key = getattr(auth, "key_id", None) if auth is not None else None
            await self.limiter.check_or_raise(key or "anonymous")
        except RAGXError as exc:
            # Middleware runs outside FastAPI's ExceptionMiddleware, so build
            # the unified error envelope directly (§9.1.2).
            return error_response(exc, getattr(request.state, "trace_id", None))
        return await call_next(request)
