"""QdrantVectorStore (11-plugins-builtin.md §11.2.2, full profile).

* Qdrant collection stores dense vectors with cosine distance
* ``supports_bm25=False`` — the retrieval layer auto-skips ``search_keyword``
  (01-spi.md §1.2); only the dense route is offered
* ``upsert`` is idempotent: a deterministic UUID (``uuid5`` of ``kb_id/chunk_id``)
  is used as the Qdrant point id so re-indexing the same chunk overwrites in place
* one collection per ``kb_id`` — physical isolation; a foreign chunk is rejected
  at write time with ``PluginContractError``
* ``FilterExpr`` translated to a Qdrant ``Filter`` (``must``/``must_not``/``should``);
  real columns (page, doc_id, ...) are stored as top-level payload fields,
  arbitrary fields are reached via ``metadata.<field>``
* the reported dense score is the **real cosine similarity** (06-retrieval.md
  §6.3.2) — Qdrant's ``search`` with ``with_vectors=True`` returns the stored
  vector; we recompute cosine so the score semantics are stable across stores

Third-party exceptions are translated at the plugin boundary (02-core.md §2.3).
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from ragx.core.exceptions import (
    PluginContractError,
    PluginTimeoutError,
    StoreUnavailableError,
)
from ragx.core.hashing import cosine_similarity
from ragx.core.models import Chunk, EmbeddedChunk, FilterExpr, ScoredChunk
from ragx.spi.interfaces import VectorStoreCapabilities

_NAME = "qdrant"

#: Top-level (non-metadata) payload fields available for filtering.
_REAL_COLUMNS: frozenset[str] = frozenset(
    {"chunk_id", "doc_id", "kb_id", "token_count", "version", "edited", "page"}
)

#: A fixed UUID namespace so point ids are deterministic across restarts.
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d")


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name) or "default"


def _qdrant_field(field: str) -> str:
    """Map a FilterExpr field to its Qdrant payload key."""
    if field in _REAL_COLUMNS:
        return field
    return f"metadata.{field}"


def _point_id(kb_id: str, chunk_id: str) -> str:
    """Deterministic UUID5 from (kb_id, chunk_id) — idempotent upsert key."""
    return str(uuid.uuid5(_NAMESPACE, f"{kb_id}/{chunk_id}"))


class QdrantVectorStore:
    """SPI ``VectorStore`` backed by Qdrant (dense-only, no BM25)."""

    name: str = _NAME
    capabilities: VectorStoreCapabilities = VectorStoreCapabilities(
        supports_bm25=False, supports_filter=True
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.url: str = str(cfg.get("url", "http://localhost:6333"))
        self.collection: str = str(cfg.get("collection", "ragx"))
        self.dimension: int = int(cfg.get("dim", 256))
        self.kb_id: str = str(cfg.get("kb_id", "default"))
        self._collection = f"{_sanitize(self.collection)}_{_sanitize(self.kb_id)}"
        self._client: Any = None  # AsyncQdrantClient (lazy)

    # -- client / collection management -------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PluginContractError(
                "qdrant-client package not installed (pip install ragx[qdrant])",
                details={"error": str(exc)},
            ) from exc
        self._client = AsyncQdrantClient(url=self.url)
        return self._client

    async def _ensure_collection(self) -> None:
        client = self._ensure_client()
        try:
            from qdrant_client.models import (  # type: ignore[import-not-found]
                Distance,
                VectorParams,
            )

            collections = await client.get_collections()
            names = {c.name for c in collections.collections}
            if self._collection in names:
                return
            await client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )
        except Exception as exc:
            raise self._translate(exc, "ensure collection") from exc

    # -- FilterExpr → Qdrant Filter ----------------------------------------
    def _clause_condition(self, clause: dict[str, Any]) -> Any:
        """One FilterExpr clause → a Qdrant FieldCondition."""
        from qdrant_client.models import (  # type: ignore[import-not-found]
            FieldCondition,
            MatchAny,
            MatchText,
            MatchValue,
            Range,
        )

        field = _qdrant_field(str(clause["field"]))
        op = str(clause["op"])
        value = clause.get("value")
        if op == "eq":
            return FieldCondition(key=field, match=MatchValue(value=value))
        if op == "ne":
            # NE is returned as a FieldCondition; the caller puts it in must_not.
            return FieldCondition(key=field, match=MatchValue(value=value))
        if op in ("gt", "ge", "lt", "le"):
            symbol = {"gt": "gt", "ge": "gte", "lt": "lt", "le": "lte"}[op]
            return FieldCondition(key=field, range=Range(**{symbol: value}))
        if op == "in":
            values = value if isinstance(value, list) else [value]
            return FieldCondition(key=field, match=MatchAny(any=[str(v) for v in values]))
        if op == "contains":
            return FieldCondition(key=field, match=MatchText(text=str(value)))
        raise PluginContractError(
            "unsupported filter operator", details={"op": op, "field": field}
        )

    def _build_filter(self, expr: FilterExpr | None) -> Any | None:
        if expr is None:
            return None
        try:
            from qdrant_client.models import Filter  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PluginContractError(
                "qdrant-client not installed", details={"error": str(exc)}
            ) from exc

        must: list[Any] = []
        must_not: list[Any] = []
        should: list[Any] = []
        for clause in expr.and_ or []:
            if str(clause["op"]) == "ne":
                must_not.append(self._clause_condition(clause))
            else:
                must.append(self._clause_condition(clause))
        for clause in expr.or_ or []:
            should.append(self._clause_condition(clause))

        kwargs: dict[str, Any] = {}
        if must:
            kwargs["must"] = must
        if must_not:
            kwargs["must_not"] = must_not
        if should:
            kwargs["should"] = should
            kwargs["min_should"] = 1
        if not kwargs:
            return None
        return Filter(**kwargs)

    # -- SPI ----------------------------------------------------------------
    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        await self._ensure_collection()
        client = self._ensure_client()
        try:
            from qdrant_client.models import PointStruct  # type: ignore[import-not-found]

            points: list[Any] = []
            for chunk in chunks:
                if chunk.kb_id != self.kb_id:
                    raise PluginContractError(
                        "chunk belongs to a different knowledge base",
                        details={"expected": self.kb_id, "got": chunk.kb_id,
                                 "chunk_id": chunk.chunk_id},
                    )
                if len(chunk.vector) != self.dimension:
                    raise PluginContractError(
                        "vector dimension mismatch",
                        details={"expected": self.dimension,
                                 "got": len(chunk.vector), "chunk_id": chunk.chunk_id},
                    )
                points.append(PointStruct(
                    id=_point_id(self.kb_id, chunk.chunk_id),
                    vector=[float(x) for x in chunk.vector],
                    payload=self._payload(chunk),
                ))
            await client.upsert(collection_name=self._collection, points=points)
        except PluginContractError:
            raise
        except Exception as exc:
            raise self._translate(exc, "upsert") from exc

    async def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        client = self._ensure_client()
        try:
            from qdrant_client.models import PointIdsList  # type: ignore[import-not-found]

            ids = [_point_id(self.kb_id, cid) for cid in chunk_ids]
            await client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=ids),
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "not found" in exc_str or "doesn't exist" in exc_str:
                return
            raise self._translate(exc, "delete") from exc

    async def search_dense(
        self, vector: list[float], *, top_k: int,
        filter_expr: FilterExpr | None = None,
    ) -> list[ScoredChunk]:
        if len(vector) != self.dimension:
            raise PluginContractError(
                "query vector dimension mismatch",
                details={"expected": self.dimension, "got": len(vector)},
            )
        client = self._ensure_client()
        qfilter = self._build_filter(filter_expr)
        try:
            results = await client.search(
                collection_name=self._collection,
                query_vector=[float(x) for x in vector],
                limit=max(top_k, 1),
                query_filter=qfilter,
                with_vectors=True,
                with_payload=True,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "not found" in exc_str or "doesn't exist" in exc_str:
                return []
            raise self._translate(exc, "search_dense") from exc

        out: list[ScoredChunk] = []
        for point in results:
            payload = point.payload or {}
            stored_vec = point.vector if hasattr(point, "vector") else None
            if isinstance(stored_vec, dict):
                # Qdrant may return vectors as {name: [...]} when multiple vectors exist
                stored_vec = stored_vec.get("") or next(iter(stored_vec.values()), [])
            if stored_vec and len(stored_vec) == len(vector):
                score = cosine_similarity(vector, stored_vec)
            else:
                score = float(point.score)
            out.append(ScoredChunk(
                chunk=self._payload_to_chunk(payload),
                score=max(0.0, min(1.0, score)),
                source="dense",
            ))
        out.sort(key=lambda h: h.score, reverse=True)
        return out[: max(top_k, 0)]

    async def search_keyword(
        self, query: str, *, top_k: int,
        filter_expr: FilterExpr | None = None,
    ) -> list[ScoredChunk]:
        """BM25 is not supported; the retrieval layer skips this route."""
        return []

    # -- payload (de)serialisation ------------------------------------------
    @staticmethod
    def _payload(chunk: EmbeddedChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "kb_id": chunk.kb_id,
            "atom_ids": chunk.atom_ids,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "page": chunk.page,
            "bbox": list(chunk.bbox) if chunk.bbox else None,
            "metadata": chunk.metadata,
            "edited": bool(chunk.edited),
            "version": chunk.version,
        }

    @staticmethod
    def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
        bbox_raw = payload.get("bbox")
        bbox = tuple(bbox_raw) if isinstance(bbox_raw, (list, tuple)) else None
        return Chunk(
            chunk_id=payload.get("chunk_id", ""),
            doc_id=payload.get("doc_id", ""),
            kb_id=payload.get("kb_id", ""),
            atom_ids=list(payload.get("atom_ids") or []),
            text=payload.get("text", ""),
            token_count=int(payload.get("token_count", 0)),
            page=payload.get("page"),
            bbox=bbox,  # type: ignore[arg-type]
            metadata=dict(payload.get("metadata") or {}),
            edited=bool(payload.get("edited", False)),
            version=int(payload.get("version", 1)),
        )

    # -- exception translation ----------------------------------------------
    @staticmethod
    def _translate(exc: Exception, context: str) -> Exception:
        exc_str = str(exc).lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            return PluginTimeoutError(
                f"Qdrant {context} timed out", details={"error": str(exc)}
            )
        if "connection" in exc_str or "unreachable" in exc_str or "refused" in exc_str:
            return StoreUnavailableError(
                f"Qdrant {context}: store unreachable", code=9001,
                details={"error": str(exc)},
            )
        if "validation" in exc_str or "invalid" in exc_str:
            return PluginContractError(
                f"Qdrant {context}: contract violation", details={"error": str(exc)}
            )
        return StoreUnavailableError(
            f"Qdrant {context} failed", code=9001, details={"error": str(exc)}
        )

    # -- lifecycle ----------------------------------------------------------
    async def startup(self) -> None:
        self._ensure_client()
        await self._ensure_collection()

    async def shutdown(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    async def count(self) -> int:
        client = self._ensure_client()
        try:
            resp = await client.count(collection_name=self._collection)
            return int(resp.count)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "not found" in exc_str or "doesn't exist" in exc_str:
                return 0
            raise self._translate(exc, "count") from exc
