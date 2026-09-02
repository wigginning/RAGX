"""Unit tests for the SemanticCache (RX-LLM-02)."""

from __future__ import annotations

from typing import Any

import pytest

from ragx.core.exceptions import CacheBackendError
from ragx.core.models import Chunk, ScoredChunk
from ragx.core.settings import CacheConfig
from ragx.llm.semantic_cache import SemanticCache


class _StubEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dimension = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # deterministic, near-identical vectors for identical text; rotate
        # one coordinate for distinct text. This is enough to exercise the
        # similarity threshold logic.
        out = []
        for t in texts:
            h = hash(t) % 1000
            base = [float((h >> i) & 1) for i in range(self.dimension)]
            out.append(base)
        return out


class _StubVectorStore:
    def __init__(self) -> None:
        self.upserted: list[Any] = []
        self.search_results: list[Any] = []

    async def upsert(self, chunks: list[Any]) -> None:
        self.upserted.extend(chunks)

    async def search_dense(
        self,
        vector: list[float],
        *,
        top_k: int,
        filter_expr: Any | None = None,
    ) -> list[Any]:
        return list(self.search_results)


class _StubExactCache:
    def __init__(self) -> None:
        self.kv: dict[tuple[str, str, int], str] = {}
        self.epochs: dict[str, int] = {}
        self.invalidated: list[str] = []

    async def get(self, key: str, *, kb_id: str = "default", epoch: int = 1) -> str | None:
        return self.kv.get((kb_id, key, epoch))

    async def set(
        self,
        key: str,
        value: str,
        *,
        kb_id: str = "default",
        epoch: int = 1,
        ttl: int | None = None,
    ) -> bool:
        self.kv[(kb_id, key, epoch)] = value
        return True

    async def invalidate_kb(self, kb_id: str) -> bool:
        self.epochs.setdefault(kb_id, 1)
        self.epochs[kb_id] += 1
        self.invalidated.append(kb_id)
        # Drop the per-kb cache snapshot (epoch bump makes prior epoch unreachable).
        self.kv = {k: v for k, v in self.kv.items() if k[0] != kb_id}
        return True

    async def get_epoch(self, kb_id: str) -> int:
        return self.epochs.get(kb_id, 1)


def _cache_hit(chunk_id: str, embedding: list[float], answer: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="__semantic_cache__",
            kb_id="default",
            atom_ids=[],
            text="cached query",
            token_count=10,
            page=None,
            bbox=None,
            metadata={"cache_embedding": embedding, "cache_answer": answer},
            edited=False,
            version=1,
        ),
        score=0.99,
        source="dense",
    )


@pytest.fixture
def cache_setup() -> tuple[SemanticCache, _StubExactCache, _StubVectorStore, _StubEmbedder]:
    embedder = _StubEmbedder()
    store = _StubVectorStore()
    exact = _StubExactCache()
    cache = SemanticCache(embedder, store, exact_cache=exact, config=CacheConfig(similarity=0.9))
    return cache, exact, store, embedder


async def test_disabled_cache_returns_none() -> None:
    embedder = _StubEmbedder()
    store = _StubVectorStore()
    cache = SemanticCache(
        embedder, store, exact_cache=None, config=CacheConfig(enabled=False)
    )
    assert await cache.lookup("anything") is None


async def test_exact_tier_returns_cached_answer(cache_setup: Any) -> None:
    cache, exact, store, _embedder = cache_setup
    await cache.store("hello world", "the answer", kb_id="default")
    assert await cache.lookup("hello world") == "the answer"


async def test_invalidate_kb_drops_subsequent_hits(cache_setup: Any) -> None:
    cache, exact, store, _embedder = cache_setup
    await cache.store("query 1", "answer 1")
    assert await cache.lookup("query 1") == "answer 1"
    await cache.invalidate_kb("default")
    # After invalidation, the entry is gone (epoch bumped).
    assert await cache.lookup("query 1") is None
    assert exact.invalidated == ["default"]


async def test_similarity_threshold_filters_low_sim(cache_setup: Any) -> None:
    cache, exact, store, embedder = cache_setup
    # Manually plant a vector entry with a near-orthogonal cached vector.
    qvec = (await embedder.embed(["z"]))[0]
    far_vec = [-x for x in qvec]  # dot product ~ -norm^2 → cosine ~ -1
    store.search_results = [_cache_hit("cached_1", far_vec, "wrong answer")]
    assert await cache.lookup("z") is None


async def test_high_similarity_returns_answer(cache_setup: Any) -> None:
    cache, exact, store, embedder = cache_setup
    qvec = (await embedder.embed(["hello"]))[0]
    store.search_results = [_cache_hit("cached_1", list(qvec), "right answer")]
    out = await cache.lookup("hello")
    assert out == "right answer"


async def test_fast_only_skips_vector_tier(cache_setup: Any) -> None:
    cache, exact, store, embedder = cache_setup
    # Plant a vector hit; fast_only mode must skip it.
    cache.config = CacheConfig(similarity=0.9, fast_only=True)
    qvec = (await embedder.embed(["hi"]))[0]
    store.search_results = [_cache_hit("cached_1", list(qvec), "v answer")]
    # Exact-tier miss + fast_only -> miss (no vector lookup).
    assert await cache.lookup("hi") is None


async def test_exact_backend_failure_degrades_to_vector_tier() -> None:
    class _BrokenExact:
        async def get(self, *args: Any, **kwargs: Any) -> str:
            raise CacheBackendError("boom", code=6004)

        async def set(self, *args: Any, **kwargs: Any) -> bool:
            return False

        async def get_epoch(self, *args: Any, **kwargs: Any) -> int:
            return 1

        async def invalidate_kb(self, *args: Any, **kwargs: Any) -> bool:
            return False

    embedder = _StubEmbedder()
    store = _StubVectorStore()
    qvec = (await embedder.embed(["foo"]))[0]
    store.search_results = [_cache_hit("cached_1", list(qvec), "v answer")]
    cache = SemanticCache(
        embedder, store, exact_cache=_BrokenExact(), config=CacheConfig(similarity=0.9)
    )
    assert await cache.lookup("foo") == "v answer"


async def test_store_writes_to_both_tiers(cache_setup: Any) -> None:
    cache, exact, store, _embedder = cache_setup
    await cache.store("q", "a")
    # exact tier
    assert await cache.lookup("q") == "a"
    # vector tier persisted a synthetic chunk
    assert len(store.upserted) == 1
    md = store.upserted[0].metadata
    assert md["cache_answer"] == "a"
    assert md["cache_query"] == "q"


async def test_invalidate_kb_no_exact_returns_false() -> None:
    cache = SemanticCache(_StubEmbedder(), _StubVectorStore(), exact_cache=None)
    assert await cache.invalidate_kb("default") is False
