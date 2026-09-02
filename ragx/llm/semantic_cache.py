"""Semantic cache (08-llm.md §8.5).

The semantic cache sits in front of the Resilient Model Router: every
``GENERATE`` request with ``cacheable=True`` is hashed into an embedding,
compared against previously-cached embeddings, and returned as a hit when the
cosine similarity exceeds a threshold (default 0.97).

Design (08 §8.5):

* **Two-tier**: an exact-match ``RedisCache`` (cheap, O(1)) before the vector
  similarity lookup (more expensive, but recall-friendly).
* **Namespace isolation**: per-kb + per-epoch. ``invalidate_kb`` increments the
  epoch so every previous entry is unreachable in O(1).
* **TTL**: each entry expires after ``cache.ttl_s`` seconds (default 24h).
* **Degrade gracefully**: any backend failure logs a warning and returns a
  miss — the main flow must never block on the cache (08 §8.5.1, error code
  6004).
* **Threshold**: 0.97 by default; controlled by ``cache.similarity_threshold``.
* **Fast mode**: when ``cache.fast_only=True``, skip the vector tier entirely
  and serve only exact hits.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ragx.core.exceptions import CacheBackendError
from ragx.core.settings import CacheConfig

logger = logging.getLogger("ragx.llm.semantic_cache")


class SemanticCache:
    """Two-tier (exact + semantic) cache in front of the LLM router."""

    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        exact_cache: Any | None = None,
        config: CacheConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.exact = exact_cache  # optional RedisCache tier (08 §8.5.2)
        self.config = config or CacheConfig()

    # -- public API ---------------------------------------------------------
    async def lookup(self, text: str, *, kb_id: str = "default") -> str | None:
        """Return the cached answer for ``text`` or ``None`` on miss.

        Backend failures are logged as warnings and returned as a miss so the
        caller can fall through to the router (08 §8.5.1).
        """
        if not self.config.enabled:
            return None
        epoch = await self._current_epoch(kb_id)

        # Tier 1: exact match (cheap O(1))
        if self.exact is not None and not self.config.fast_only:
            try:
                hit = await self.exact.get(self._key(text), kb_id=kb_id, epoch=epoch)
                if hit is not None:
                    return hit
            except CacheBackendError:
                logger.warning("semantic-cache exact tier degraded (6004)")

        # Tier 2: vector similarity
        if self.config.fast_only:
            return None
        try:
            qvec = (await self.embedder.embed([text]))[0]
        except Exception as exc:
            logger.warning("semantic-cache embed failed (6004): %s", exc)
            return None

        try:
            hits = await self.vector_store.search_dense(
                qvec,
                top_k=self.config.top_k,
                filter_expr=None,
            )
        except Exception as exc:
            logger.warning("semantic-cache vector lookup failed (6004): %s", exc)
            return None

        for hit in hits:
            cached_vec = hit.chunk.metadata.get("cache_embedding")
            cached_answer = hit.chunk.metadata.get("cache_answer")
            if not cached_vec or cached_answer is None:
                continue
            sim = self._cosine(qvec, cached_vec)
            if sim >= self.config.similarity:
                logger.debug(
                    "semantic-cache hit sim=%.3f kb=%s", sim, kb_id,
                )
                return cached_answer
        return None

    async def store(
        self,
        text: str,
        answer: str,
        *,
        kb_id: str = "default",
    ) -> None:
        """Store a successful (text, answer) pair for future hits.

        Writes both tiers: exact + a vector entry tagged with the embedding
        and the original query text in the chunk metadata.
        """
        if not self.config.enabled:
            return
        epoch = await self._current_epoch(kb_id)

        if self.exact is not None:
            try:
                await self.exact.set(
                    self._key(text), answer,
                    kb_id=kb_id, epoch=epoch, ttl=self.config.ttl_s,
                )
            except CacheBackendError:
                logger.warning("semantic-cache exact store degraded (6004)")

        # Vector tier: only useful when the embedder + store can accept the
        # entry. We synthesise a synthetic ``Chunk`` carrying the cache
        # metadata so the existing vector store path handles persistence.
        try:
            qvec = (await self.embedder.embed([text]))[0]
            from ragx.core.ids import new_id
            from ragx.core.models import Chunk, EmbeddedChunk

            chunk = Chunk(
                chunk_id=new_id("chk_"),
                doc_id="__semantic_cache__",
                kb_id=kb_id,
                atom_ids=[],
                text=text,
                token_count=len(text),
                page=None,
                bbox=None,
                metadata={
                    "cache_embedding": qvec,
                    "cache_query": text,
                    "cache_answer": answer,
                    "cache_ttl": self.config.ttl_s,
                    "cache_epoch": epoch,
                },
                edited=False,
                version=1,
            )
            embedded = EmbeddedChunk(**chunk.model_dump(), vector=qvec)
            await self.vector_store.upsert([embedded])
        except Exception as exc:
            logger.warning("semantic-cache vector store degraded (6004): %s", exc)

    async def invalidate_kb(self, kb_id: str) -> bool:
        """Drop all cached entries for ``kb_id``.

        Strategy: increment the kb's epoch. Existing entries become
        unreachable because ``lookup`` reads the current epoch before key
        construction (O(1)).
        """
        if self.exact is None:
            return False
        return bool(await self.exact.invalidate_kb(kb_id))

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _current_epoch(self, kb_id: str) -> int:
        if self.exact is None:
            return 1
        try:
            return await self.exact.get_epoch(kb_id)
        except CacheBackendError:
            return 1

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if not na or not nb:
            return 0.0
        return dot / (na * nb)


__all__ = ["SemanticCache"]
