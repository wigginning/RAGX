"""ESVectorStore (11-plugins-builtin.md §11.2.1, full profile).

* Elasticsearch ≥ 8.11; ``dense_vector`` field with cosine similarity + ``knn``
  query for dense retrieval; built-in ``match`` + BM25 for keyword retrieval —
  both in a single index (the design mandates "dense 与 BM25 一体")
* ``upsert`` is idempotent: ES ``_id = chunk_id`` so re-indexing the same chunk
  overwrites in place (03-ingestion.md §3.6.1)
* one index per ``kb_id`` (``{index_prefix}_{kb_id}``) — physical isolation so
  a foreign chunk is rejected at write time with ``PluginContractError``
* ``FilterExpr`` translated to an ES ``bool`` filter; real columns (page, doc_id,
  ...) are indexed directly, arbitrary fields are reached via ``metadata.<field>``
* the reported dense score is the **real cosine similarity** (06-retrieval.md
  §6.3.2) — ES ``knn`` returns candidates; we fetch the stored vector from
  ``_source`` and recompute cosine so the retrieval layer's score semantics are
  stable across stores
* BM25 scores are normalised to ``[0, 1]`` via ``1 / (1 + max(0, bm25))`` so they
  are comparable to the dense route's ``[0, 1]`` range before RRF fusion

Third-party exceptions (``elasticsearch.*``) are translated at the plugin
boundary to the RAGX exception tree (02-core.md §2.3); raw exceptions never
cross the SPI.
"""

from __future__ import annotations

import re
from typing import Any

from ragx.core.exceptions import (
    PluginContractError,
    PluginTimeoutError,
    StoreUnavailableError,
)
from ragx.core.hashing import cosine_similarity
from ragx.core.models import Chunk, EmbeddedChunk, FilterExpr, ScoredChunk
from ragx.spi.interfaces import VectorStoreCapabilities

_NAME = "es"

#: Top-level (non-metadata) columns indexed for filtering.
_REAL_COLUMNS: frozenset[str] = frozenset(
    {"chunk_id", "doc_id", "kb_id", "token_count", "version", "edited", "page"}
)

#: ES field type per real column (used in mapping + filter pushdown).
_REAL_TYPES: dict[str, str] = {
    "chunk_id": "keyword",
    "doc_id": "keyword",
    "kb_id": "keyword",
    "token_count": "integer",
    "version": "integer",
    "edited": "boolean",
    "page": "integer",
}


def _sanitize(name: str) -> str:
    """Make a string safe for an ES index name suffix."""
    return re.sub(r"[^0-9A-Za-z_]", "_", name) or "default"


def _es_field(field: str) -> str:
    """Map a FilterExpr field to its ES document path."""
    if field in _REAL_COLUMNS:
        return field
    return f"metadata.{field}"


class ESVectorStore:
    """SPI ``VectorStore`` backed by Elasticsearch (dense + BM25 in one index)."""

    name: str = _NAME
    capabilities: VectorStoreCapabilities = VectorStoreCapabilities(
        supports_bm25=True, supports_filter=True
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.hosts: list[str] = list(cfg.get("hosts", ["http://localhost:9200"]))
        self.index_prefix: str = str(cfg.get("index_prefix", "ragx"))
        self.dimension: int = int(cfg.get("dim", 256))
        self.kb_id: str = str(cfg.get("kb_id", "default"))
        self._index = f"{self.index_prefix}_{_sanitize(self.kb_id)}"
        self._client: Any = None  # AsyncElasticsearch (lazy)

    # -- client / index management ------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from elasticsearch import AsyncElasticsearch
        except ImportError as exc:
            raise PluginContractError(
                "elasticsearch package not installed (pip install ragx[es])",
                details={"error": str(exc)},
            ) from exc
        self._client = AsyncElasticsearch(hosts=self.hosts)
        return self._client

    def _mapping(self) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for col, typ in _REAL_TYPES.items():
            props[col] = {"type": typ}
        props["atom_ids"] = {"type": "keyword"}
        props["text"] = {"type": "text"}
        props["bbox"] = {"type": "object", "enabled": False}
        props["metadata"] = {"type": "object", "enabled": True}
        props["vector"] = {
            "type": "dense_vector",
            "dims": self.dimension,
            "index": True,
            "similarity": "cosine",
        }
        return {"mappings": {"properties": props}}

    async def _ensure_index(self) -> None:
        client = self._ensure_client()
        try:
            exists = await client.indices.exists(index=self._index)
            if exists:
                return
            await client.indices.create(index=self._index, **self._mapping())
        except Exception as exc:
            raise self._translate(exc, "ensure index") from exc

    # -- FilterExpr → ES bool filter ----------------------------------------
    def _clause_filter(self, clause: dict[str, Any]) -> dict[str, Any]:
        """One FilterExpr clause → an ES query clause (bool filter dict)."""
        field = _es_field(str(clause["field"]))
        op = str(clause["op"])
        value = clause.get("value")
        if op == "eq":
            return {"term": {field: {"value": value}}}
        if op == "ne":
            return {"bool": {"must_not": [{"term": {field: {"value": value}}}]}}
        if op in ("gt", "ge", "lt", "le"):
            symbol = {"gt": "gt", "ge": "gte", "lt": "lt", "le": "lte"}[op]
            return {"range": {field: {symbol: value}}}
        if op == "in":
            values = value if isinstance(value, list) else [value]
            return {"terms": {field: [str(v) for v in values]}}
        if op == "contains":
            return {"wildcard": {field: {"value": f"*{str(value)}*", "case_insensitive": True}}}
        raise PluginContractError(
            "unsupported filter operator", details={"op": op, "field": field}
        )

    def _group_filter(
        self, clauses: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        return [self._clause_filter(c) for c in clauses] if clauses else []

    def _build_filter(self, expr: FilterExpr | None) -> list[dict[str, Any]]:
        """FilterExpr → list of ES filter clauses (AND semantics)."""
        if expr is None:
            return []
        and_clauses = self._group_filter(expr.and_)
        or_clauses = self._group_filter(expr.or_)
        if or_clauses:
            # OR group wrapped in a should clause with minimum_should_match=1
            return [*and_clauses, {"bool": {"should": or_clauses, "minimum_should_match": 1}}]
        return and_clauses

    # -- SPI ----------------------------------------------------------------
    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return
        await self._ensure_index()
        client = self._ensure_client()
        actions: list[dict[str, Any]] = []
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
            actions.append({
                "_op_type": "index",
                "_index": self._index,
                "_id": chunk.chunk_id,
                "_source": self._doc(chunk),
            })

        try:
            from elasticsearch.helpers import async_bulk

            await async_bulk(client, actions, raise_on_error=True)
            # Make writes visible to immediately-following searches
            # (read-your-writes; mirrors delete()'s refresh=True).
            await client.indices.refresh(index=self._index)
        except Exception as exc:
            raise self._translate(exc, "bulk upsert") from exc

    async def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        client = self._ensure_client()
        try:
            await client.delete_by_query(
                index=self._index,
                body={"query": {"terms": {"chunk_id": chunk_ids}}},
                refresh=True,
            )
        except Exception as exc:
            # If the index doesn't exist yet, there's nothing to delete — not an error.
            if "index_not_found" in str(exc).lower() or "not found" in str(exc).lower():
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
        es_filter = self._build_filter(filter_expr)
        knn: dict[str, Any] = {
            "field": "vector",
            "query_vector": [float(x) for x in vector],
            "k": max(top_k, 1),
            "num_candidates": max(top_k * 10, 100),
        }
        if es_filter:
            knn["filter"] = es_filter
        body: dict[str, Any] = {
            "knn": knn,
            "_source": True,
            "size": max(top_k, 0),
        }
        try:
            resp = await client.search(index=self._index, body=body)
        except Exception as exc:
            if "index_not_found" in str(exc).lower():
                return []
            raise self._translate(exc, "search_dense") from exc

        out: list[ScoredChunk] = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            vec = src.get("vector") or []
            if vec and len(vec) == len(vector):
                score = cosine_similarity(vector, vec)
            else:
                # Fallback: ES cosine _score = (1 + cos) / 2 → cos = 2*_score - 1
                score = max(0.0, 2.0 * float(hit.get("_score", 0.0)) - 1.0)
            out.append(ScoredChunk(
                chunk=self._source_to_chunk(src),
                score=max(0.0, min(1.0, score)),
                source="dense",
            ))
        out.sort(key=lambda h: h.score, reverse=True)
        return out[: max(top_k, 0)]

    async def search_keyword(
        self, query: str, *, top_k: int,
        filter_expr: FilterExpr | None = None,
    ) -> list[ScoredChunk]:
        if not self.capabilities.supports_bm25:
            return []
        client = self._ensure_client()
        es_filter = self._build_filter(filter_expr)
        must: list[dict[str, Any]] = [{"match": {"text": {"query": query}}}]
        bool_q: dict[str, Any] = {"must": must}
        if es_filter:
            bool_q["filter"] = es_filter
        body: dict[str, Any] = {
            "query": {"bool": bool_q},
            "_source": True,
            "size": max(top_k, 0),
        }
        try:
            resp = await client.search(index=self._index, body=body)
        except Exception as exc:
            if "index_not_found" in str(exc).lower():
                return []
            raise self._translate(exc, "search_keyword") from exc

        out: list[ScoredChunk] = []
        for hit in resp.get("hits", {}).get("hits", []):
            src = hit.get("_source", {})
            raw_bm25 = float(hit.get("_score", 0.0))
            # ES BM25 _score is >= 0; normalise to [0, 1) via s/(1+s).
            score = raw_bm25 / (1.0 + raw_bm25) if raw_bm25 > 0 else 0.0
            out.append(ScoredChunk(
                chunk=self._source_to_chunk(src),
                score=score,
                source="bm25",
            ))
        return out

    # -- document (de)serialisation ----------------------------------------
    @staticmethod
    def _doc(chunk: EmbeddedChunk) -> dict[str, Any]:
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
            "vector": [float(x) for x in chunk.vector],
        }

    @staticmethod
    def _source_to_chunk(src: dict[str, Any]) -> Chunk:
        bbox_raw = src.get("bbox")
        bbox = tuple(bbox_raw) if isinstance(bbox_raw, (list, tuple)) else None
        return Chunk(
            chunk_id=src.get("chunk_id", ""),
            doc_id=src.get("doc_id", ""),
            kb_id=src.get("kb_id", ""),
            atom_ids=list(src.get("atom_ids") or []),
            text=src.get("text", ""),
            token_count=int(src.get("token_count", 0)),
            page=src.get("page"),
            bbox=bbox,  # type: ignore[arg-type]
            metadata=dict(src.get("metadata") or {}),
            edited=bool(src.get("edited", False)),
            version=int(src.get("version", 1)),
        )

    # -- exception translation ----------------------------------------------
    @staticmethod
    def _translate(exc: Exception, context: str) -> Exception:
        """Map third-party ES exceptions to the RAGX exception tree."""
        exc_str = str(exc).lower()
        if "timeout" in exc_str or "timed out" in exc_str:
            return PluginTimeoutError(
                f"ES {context} timed out", details={"error": str(exc)}
            )
        if "connection" in exc_str or "unreachable" in exc_str or "transport" in exc_str:
            return StoreUnavailableError(
                f"ES {context}: store unreachable", code=9001,
                details={"error": str(exc)},
            )
        if "mapper_parsing" in exc_str or "illegal_argument" in exc_str:
            return PluginContractError(
                f"ES {context}: contract violation", details={"error": str(exc)}
            )
        return StoreUnavailableError(
            f"ES {context} failed", code=9001, details={"error": str(exc)}
        )

    # -- lifecycle ----------------------------------------------------------
    async def startup(self) -> None:
        self._ensure_client()
        await self._ensure_index()

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
            resp = await client.count(index=self._index)
            return int(resp.get("count", 0))
        except Exception as exc:
            if "index_not_found" in str(exc).lower():
                return 0
            raise self._translate(exc, "count") from exc
