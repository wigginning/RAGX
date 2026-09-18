"""RedisCache unit tests (Agent D RX-RET-06).

The contract for RedisCache is graceful degradation (08-llm.md §8.5.1): when
Redis is unreachable, every read returns ``None`` and every write returns
``False`` — it must never raise, so the main flow is never blocked.

The actual get/set round-trip is covered by integration tests against a real
Redis (``deploy/compose/test.yml`` service ``redis``); here we only pin the
fail-open behaviour on an unreachable endpoint plus the key-namespace layout.
"""

from __future__ import annotations

from ragx.plugins.cache_redis import RedisCache


def test_key_layout() -> None:
    """Keys are namespaced as ``{ns}:{kb}:{epoch}:{key}`` for O(1) invalidation."""
    cache = RedisCache({"namespace": "ragx_semantic_cache", "ttl_s": 60})
    full = cache._full_key("kb_1", 3, "hello")
    assert full == "ragx_semantic_cache:kb_1:3:hello"
    assert cache.ttl_s == 60


async def test_get_degrades_to_miss_on_unreachable() -> None:
    """Redis down -> get returns None (miss), never raises."""
    cache = RedisCache({"url": "redis://127.0.0.1:1/0", "namespace": "ns"})
    assert await cache.get("key", kb_id="kb") is None
    assert await cache.get_many(["a", "b"], kb_id="kb") == [None, None]


async def test_set_degrades_to_noop_on_unreachable() -> None:
    """Redis down -> set returns False (no-op), never raises."""
    cache = RedisCache({"url": "redis://127.0.0.1:1/0", "namespace": "ns"})
    assert await cache.set("key", "value", kb_id="kb") is False
    assert await cache.set_many([("a", "1")], kb_id="kb") is False


async def test_invalidate_degrades_to_false_on_unreachable() -> None:
    """Redis down -> invalidate_kb returns False, never raises."""
    cache = RedisCache({"url": "redis://127.0.0.1:1/0", "namespace": "ns"})
    assert await cache.invalidate_kb("kb") is False
    assert await cache.get_epoch("kb") == 1
