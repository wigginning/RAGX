"""In-memory exact cache (11-plugins-builtin.md §11.2.9, lite sibling).

Mirrors :class:`~ragx.plugins.cache_redis.RedisCache`'s exact-match tier
interface so the :class:`~ragx.llm.semantic_cache.SemanticCache` can run in the
lite profile without a Redis dependency.

Design points (kept identical to RedisCache for drop-in swapping):

* keys are namespaced as ``{namespace}:{kb_id}:{epoch}:{key}`` so that
  ``invalidate_kb`` is O(1) — a single epoch increment makes all prior entries
  unreachable (08-llm.md §8.5.3).
* TTL is enforced lazily on ``get`` (epoch/ttl stored in-process).
* graceful degradation: a backend failure can never happen here (it's local
  memory), but we still return ``None``/``False`` on bad input rather than
  raising — the cache must never block the main flow (08-llm.md §8.5.1).
"""

from __future__ import annotations

import logging
import time
from typing import Any

_NAME = "memory"
logger = logging.getLogger("ragx.plugins.cache_memory")


class InMemoryExactCache:
    """A TTL key-value cache backed by an in-process dict (lite profile)."""

    name: str = _NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.namespace: str = str(cfg.get("namespace", "ragx_semantic_cache"))
        self.ttl_s: int = int(cfg.get("ttl_s", 86_400))
        #: full_key -> (expires_at_epoch_seconds, value)
        self._entries: dict[str, tuple[float, str]] = {}
        #: kb_id -> current epoch
        self._epochs: dict[str, int] = {}

    # -- client (no-op; local memory) ---------------------------------------
    def _full_key(self, kb_id: str, epoch: int, key: str) -> str:
        return f"{self.namespace}:{kb_id}:{epoch}:{key}"

    # -- public API ---------------------------------------------------------
    async def get(
        self, key: str, *, kb_id: str = "default", epoch: int = 1
    ) -> str | None:
        """Return the cached value or ``None`` (miss or expired)."""
        full_key = self._full_key(kb_id, epoch, key)
        entry = self._entries.get(full_key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at is not None and time.time() > expires_at:
            self._entries.pop(full_key, None)
            return None
        return value

    async def set(
        self,
        key: str,
        value: str,
        *,
        kb_id: str = "default",
        epoch: int = 1,
        ttl: int | None = None,
    ) -> bool:
        """Store a value with TTL; returns ``True`` (always succeeds locally)."""
        full_key = self._full_key(kb_id, epoch, key)
        effective_ttl = ttl if ttl is not None else self.ttl_s
        try:
            self._entries[full_key] = (time.time() + effective_ttl, value)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("InMemoryExactCache.set failed: %s", exc)
            return False

    async def delete(
        self, key: str, *, kb_id: str = "default", epoch: int = 1
    ) -> None:
        self._entries.pop(self._full_key(kb_id, epoch, key), None)

    async def get_many(
        self,
        keys: list[str],
        *,
        kb_id: str = "default",
        epoch: int = 1,
    ) -> list[str | None]:
        """Batch get; each element is the value or ``None`` on miss."""
        return [await self.get(k, kb_id=kb_id, epoch=epoch) for k in keys]

    async def set_many(
        self,
        items: list[tuple[str, str]],
        *,
        kb_id: str = "default",
        epoch: int = 1,
        ttl: int | None = None,
    ) -> bool:
        """Batch set; returns ``True`` (always succeeds locally)."""
        for key, value in items:
            await self.set(key, value, kb_id=kb_id, epoch=epoch, ttl=ttl)
        return True

    async def get_epoch(self, kb_id: str) -> int:
        """Return the current cache epoch for a kb (1 if unset)."""
        return self._epochs.get(kb_id, 1)

    async def invalidate_kb(self, kb_id: str) -> bool:
        """Epoch-based O(1) invalidation — increment the kb's epoch counter.

        Lazy-drops entries belonging to the now-stale epoch to free memory.
        """
        new_epoch = self._epochs.get(kb_id, 1) + 1
        self._epochs[kb_id] = new_epoch
        prefix = f"{self.namespace}:{kb_id}:"
        cur_prefix = f"{prefix}{new_epoch}:"
        for full_key in [
            k
            for k in self._entries
            if k.startswith(prefix) and not k.startswith(cur_prefix)
        ]:
            self._entries.pop(full_key, None)
        return True

    # -- lifecycle (no-ops for local memory) --------------------------------
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        self._entries.clear()
        self._epochs.clear()

    async def ping(self) -> bool:
        """Health check (used by /v1/health?deep=true)."""
        return True


__all__ = ["InMemoryExactCache"]
