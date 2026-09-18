"""Unit tests for InMemoryExactCache (LLM-02 lite exact tier)."""

from __future__ import annotations

import pytest

from ragx.plugins.cache_memory import InMemoryExactCache


def _cache(**kw: object) -> InMemoryExactCache:
    return InMemoryExactCache({"namespace": "ns", "ttl_s": 100})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_set_get_roundtrip() -> None:
    c = _cache()
    assert await c.set("k", "v", kb_id="kb1", epoch=1) is True
    assert await c.get("k", kb_id="kb1", epoch=1) == "v"


@pytest.mark.asyncio
async def test_namespace_isolation() -> None:
    c = _cache()
    await c.set("k", "v", kb_id="kb1", epoch=1)
    # different kb / epoch => miss
    assert await c.get("k", kb_id="kb2", epoch=1) is None
    assert await c.get("k", kb_id="kb1", epoch=2) is None


@pytest.mark.asyncio
async def test_ttl_expiry() -> None:
    c = _cache()
    await c.set("k", "v", kb_id="kb1", epoch=1, ttl=0)
    # ttl=0 -> expires immediately
    assert await c.get("k", kb_id="kb1", epoch=1) is None


@pytest.mark.asyncio
async def test_epoch_invalidation_drops_old_entries() -> None:
    c = _cache()
    await c.set("k", "v1", kb_id="kb1", epoch=1)
    assert await c.get("k", kb_id="kb1", epoch=1) == "v1"
    # invalidate_kb bumps the epoch -> old entry unreachable
    assert await c.invalidate_kb("kb1") is True
    assert await c.get("k", kb_id="kb1", epoch=1) is None
    # new epoch is reachable for fresh writes
    await c.set("k", "v2", kb_id="kb1", epoch=2)
    assert await c.get("k", kb_id="kb1", epoch=2) == "v2"


@pytest.mark.asyncio
async def test_get_epoch_defaults_to_one() -> None:
    c = _cache()
    assert await c.get_epoch("fresh") == 1


@pytest.mark.asyncio
async def test_get_many_and_set_many() -> None:
    c = _cache()
    await c.set_many([("a", "1"), ("b", "2")], kb_id="kb1", epoch=1)
    vals = await c.get_many(["a", "b", "c"], kb_id="kb1", epoch=1)
    assert vals == ["1", "2", None]


@pytest.mark.asyncio
async def test_lifecycle_noops() -> None:
    c = _cache()
    await c.startup()
    assert await c.ping() is True
    await c.shutdown()  # clears entries
    await c.set("k", "v", kb_id="kb1", epoch=1)
    assert await c.get("k", kb_id="kb1", epoch=1) == "v"
