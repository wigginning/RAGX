"""RedisCache (infra component, parallels RedisQueue — 11-plugins-builtin.md §11.2.9).

Not part of the SPI seven interfaces; it is a cross-cutting storage dependency
used by:

* the semantic cache (08-llm.md §8.5) — as a fast **exact-match** tier before
  the vector-similarity lookup in ``VectorStore``
* the topic-summary cache (05-kg.md §5.5.2, ``topic_cache``)
* any downstream component that needs a TTL key-value store

Design points:

* keys are namespaced as ``{namespace}:{kb_id}:{epoch}:{key}`` so that
  ``invalidate_kb`` is O(1) — a single epoch increment makes all prior entries
  unreachable (08-llm.md §8.5.3). Lazy TTL expiry is left to Redis.
* graceful degradation: on Redis failure the cache logs a warning and returns
  ``None`` / ``False`` — **never** blocks the main flow (08-llm.md §8.5.1:
  "缓存查询/写入失败 → 记 6004 警告 span，不阻断主流程").
"""

from __future__ import annotations

import logging
from typing import Any

from ragx.core.exceptions import CacheBackendError

_NAME = "redis"
logger = logging.getLogger("ragx.plugins.cache_redis")

#: Lua script for atomic epoch-based key scan + delete (namespace invalidation).
_INVALIDATE_LUA = """
local prefix = ARGV[1]
local cursor = '0'
repeat
    local reply = redis.call('SCAN', cursor, 'MATCH', prefix .. ':*', 'COUNT', 500)
    cursor = reply[1]
    local keys = reply[2]
    if #keys > 0 then
        redis.call('DEL', unpack(keys))
    end
until cursor == '0'
return 1
"""


class RedisCache:
    """A TTL key-value cache backed by Redis with namespace isolation."""

    name: str = _NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.url: str = str(cfg.get("url", "redis://localhost:6379/0"))
        self.namespace: str = str(cfg.get("namespace", "ragx_semantic_cache"))
        self.ttl_s: int = int(cfg.get("ttl_s", 86_400))
        self._client: Any = None  # redis.asyncio.Redis (lazy)
        self._invalidate_sha: str | None = None

    # -- client -------------------------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise CacheBackendError(
                "redis package not installed (pip install ragx[redis])",
                code=6004, details={"error": str(exc)},
            ) from exc
        self._client = aioredis.from_url(
            self.url, decode_responses=True, socket_timeout=5.0,
        )
        return self._client

    def _full_key(self, kb_id: str, epoch: int, key: str) -> str:
        return f"{self.namespace}:{kb_id}:{epoch}:{key}"

    # -- public API ---------------------------------------------------------
    async def get(self, key: str, *, kb_id: str = "default", epoch: int = 1) -> str | None:
        """Return the cached value or ``None`` (miss or backend error)."""
        client = self._ensure_client()
        full_key = self._full_key(kb_id, epoch, key)
        try:
            return await client.get(full_key)
        except Exception as exc:
            logger.warning("RedisCache.get failed (degrading to miss): %s", exc)
            return None

    async def set(
        self, key: str, value: str, *,
        kb_id: str = "default", epoch: int = 1, ttl: int | None = None,
    ) -> bool:
        """Store a value with TTL; returns ``False`` on backend failure."""
        client = self._ensure_client()
        full_key = self._full_key(kb_id, epoch, key)
        effective_ttl = ttl if ttl is not None else self.ttl_s
        try:
            await client.setex(full_key, effective_ttl, value)
            return True
        except Exception as exc:
            logger.warning("RedisCache.set failed (degrading to no-op): %s", exc)
            return False

    async def delete(self, key: str, *, kb_id: str = "default", epoch: int = 1) -> None:
        client = self._ensure_client()
        full_key = self._full_key(kb_id, epoch, key)
        try:
            await client.delete(full_key)
        except Exception as exc:
            logger.warning("RedisCache.delete failed (no-op): %s", exc)

    async def get_many(
        self, keys: list[str], *, kb_id: str = "default", epoch: int = 1,
    ) -> list[str | None]:
        """Batch get; each element is the value or ``None`` on miss/error."""
        if not keys:
            return []
        client = self._ensure_client()
        full_keys = [self._full_key(kb_id, epoch, k) for k in keys]
        try:
            values = await client.mget(full_keys)
            # Redis returns None for missing keys, which is exactly what we want.
            return [v if isinstance(v, str) else None for v in values]
        except Exception as exc:
            logger.warning("RedisCache.get_many failed: %s", exc)
            return [None] * len(keys)

    async def set_many(
        self, items: list[tuple[str, str]], *,
        kb_id: str = "default", epoch: int = 1, ttl: int | None = None,
    ) -> bool:
        """Batch set via pipeline; returns ``False`` on backend failure."""
        if not items:
            return True
        client = self._ensure_client()
        effective_ttl = ttl if ttl is not None else self.ttl_s
        try:
            pipe = client.pipeline()
            for key, value in items:
                pipe.setex(self._full_key(kb_id, epoch, key), effective_ttl, value)
            await pipe.execute()
            return True
        except Exception as exc:
            logger.warning("RedisCache.set_many failed: %s", exc)
            return False

    async def invalidate_kb(self, kb_id: str) -> bool:
        """Epoch-based O(1) invalidation — increment the kb's epoch counter.

        Stores ``{namespace}:epoch:{kb_id}`` as an integer in Redis; the
        effective epoch is the max of the stored value and 1. Callers read
        the current epoch before constructing cache keys.
        """
        client = self._ensure_client()
        epoch_key = f"{self.namespace}:epoch:{kb_id}"
        try:
            new_epoch = await client.incr(epoch_key)
            # Best-effort: also lazily delete old-epoch keys to free memory.
            # This is non-blocking; stale entries are unreachable anyway.
            old_prefix = f"{self.namespace}:{kb_id}:{new_epoch - 1}"
            await self._scan_delete(old_prefix)
            return True
        except Exception as exc:
            logger.warning("RedisCache.invalidate_kb failed: %s", exc)
            return False

    async def get_epoch(self, kb_id: str) -> int:
        """Return the current cache epoch for a kb (1 if unset)."""
        client = self._ensure_client()
        epoch_key = f"{self.namespace}:epoch:{kb_id}"
        try:
            val = await client.get(epoch_key)
            return int(val) if val else 1
        except Exception:
            return 1

    async def _scan_delete(self, prefix: str) -> None:
        """Lazy deletion of all keys matching a prefix (best-effort)."""
        client = self._ensure_client()
        try:
            if self._invalidate_sha is None:
                self._invalidate_sha = await client.script_load(_INVALIDATE_LUA)
            await client.evalsha(self._invalidate_sha, 0, prefix)
        except Exception as exc:
            logger.debug("RedisCache._scan_delete skipped: %s", exc)

    # -- lifecycle ----------------------------------------------------------
    async def startup(self) -> None:
        self._ensure_client()

    async def shutdown(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    async def ping(self) -> bool:
        """Health check (used by /v1/health?deep=true)."""
        client = self._ensure_client()
        try:
            return bool(await client.ping())
        except Exception:
            return False
