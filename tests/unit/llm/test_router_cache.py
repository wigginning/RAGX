"""LLM-02 runtime wiring: ResilientRouter consults/stores the SemanticCache.

The cache *class* is covered in ``test_semantic_cache.py``; this module locks
in that the router actually uses an injected cache (RX-LLM-02 DoD: similar
query hit, cache store on miss, no store when disabled / non-cacheable).
"""

from __future__ import annotations

from typing import Any

import pytest

from ragx.core.roles import LLMRole
from ragx.core.settings import CacheConfig, LLMRouterConfig, ProviderTarget, RoleConfig
from ragx.llm.router import ResilientRouter

from .test_router import _registry, _req  # reuse fakes


class _FakeCache:
    """Records store calls; returns a primed value on lookup when set."""

    def __init__(self) -> None:
        self.store_calls: list[tuple[str, str, str]] = []
        self.lookup_return: str | None = None

    async def lookup(self, text: str, *, kb_id: str = "default") -> str | None:
        return self.lookup_return

    async def store(self, text: str, answer: str, *, kb_id: str = "default") -> None:
        self.store_calls.append((text, answer, kb_id))

    async def invalidate_kb(self, kb_id: str) -> bool:
        return True


def _cached_router(cache: Any, cache_cfg: CacheConfig | None = None) -> ResilientRouter:
    cfg = LLMRouterConfig(
        roles={
            LLMRole.GENERATE: RoleConfig(
                primary=ProviderTarget(provider="a", model="m"),
                max_retries=1,
            )
        },
        cache=cache_cfg or CacheConfig(),
    )
    reg = _registry({"a": []})
    return ResilientRouter(cfg, reg, cache=cache)


@pytest.mark.asyncio
async def test_cache_store_on_miss() -> None:
    """A cacheable miss calls the provider and writes the answer through."""
    cache = _FakeCache()
    router = _cached_router(cache)
    resp = await router.chat(_req(cacheable=True))
    assert resp.text == "mock answer"
    assert len(cache.store_calls) == 1


@pytest.mark.asyncio
async def test_cache_hit_short_circuits_provider() -> None:
    """A primed lookup returns the cached answer and skips the provider/store."""
    cache = _FakeCache()
    router = _cached_router(cache)
    await router.chat(_req(cacheable=True))  # miss -> store
    assert len(cache.store_calls) == 1

    cache.lookup_return = "CACHED ANSWER"
    resp = await router.chat(_req(cacheable=True))  # hit
    assert resp.text == "CACHED ANSWER"
    assert resp.cached is True
    # provider not re-invoked, answer not stored a second time
    assert len(cache.store_calls) == 1


@pytest.mark.asyncio
async def test_non_cacheable_skips_cache() -> None:
    cache = _FakeCache()
    router = _cached_router(cache)
    await router.chat(_req(cacheable=False))
    assert len(cache.store_calls) == 0


@pytest.mark.asyncio
async def test_disabled_cache_skips_store() -> None:
    cache = _FakeCache()
    router = _cached_router(cache, cache_cfg=CacheConfig(enabled=False))
    await router.chat(_req(cacheable=True))
    assert len(cache.store_calls) == 0
