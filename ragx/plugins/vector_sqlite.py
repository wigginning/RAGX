"""SQLiteVectorStore (11-plugins-builtin.md §11.1.2, lite profile).

* dense vectors as BLOBs, brute-force cosine (``supports_bm25`` + dense in one
  engine; single file, zero external services)
* BM25 via a SQLite FTS5 external-content table kept in sync by triggers
* ``FilterExpr`` translated to SQL over the ``metadata`` JSON column
* ``upsert`` is idempotent on ``chunk_id`` (03-ingestion.md §3.6.1)
* instance is scoped to one ``kb_id`` - foreign chunks are rejected with
  ``PluginContractError`` so kb isolation cannot be violated

Deviation from the doc: the design names ``sqlite-vec`` for KNN acceleration.
It is not available in the target environment, so dense search is exact
brute-force - correct at the lite scale the doc specifies ("万级 Chunk 以下").

Second deviation: the FTS5 table uses the ``trigram`` tokenizer. The default
``unicode61`` tokenizer does not segment CJK text, so Chinese queries would
never match. Trigram indexes every 3-character substring, giving substring
recall for both CJK and ASCII at the cost of a >=3-char minimum query length
(acceptable for the lite profile; the retrieval layer treats empty recall as a
soft miss).
"""

from __future__ import annotations

import array
import asyncio
import json
import re
import sqlite3
import threading
from typing import Any

from ragx.core.exceptions import PluginContractError
from ragx.core.hashing import cosine_similarity
from ragx.core.models import Chunk, EmbeddedChunk, FilterExpr, ScoredChunk
from ragx.spi.interfaces import VectorStoreCapabilities

_NAME = "sqlite"
_COLUMNS = (
    "chunk_id", "doc_id", "kb_id", "atom_ids", "text", "token_count",
    "page", "bbox", "metadata", "edited", "version",
)
_REAL_COLUMNS = {"page", "doc_id", "kb_id", "token_count", "version", "edited", "chunk_id"}
_SCANNABLE_FIELDS = {"page", "token_count", "version"}


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name) or "default"


def _vec_to_blob(vec: list[float]) -> bytes:
    return array.array("d", vec).tobytes()


def _blob_to_vec(blob: bytes) -> list[float]:
    return list(array.array("d", blob))


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        kb_id=row["kb_id"],
        atom_ids=json.loads(row["atom_ids"] or "[]"),
        text=row["text"],
        token_count=row["token_count"],
        page=row["page"],
        bbox=json.loads(row["bbox"]) if row["bbox"] else None,
        metadata=json.loads(row["metadata"] or "{}"),
        edited=bool(row["edited"]),
        version=row["version"],
    )


def _quote(text: str) -> str:
    """Escape a user query for FTS5 double-quoted phrase syntax."""
    return '"{}"'.format(text.replace('"', '""'))


class SQLiteVectorStore:
    """SPI ``VectorStore`` backed by a single SQLite database file."""

    name: str = _NAME
    capabilities: VectorStoreCapabilities = VectorStoreCapabilities(
        supports_bm25=True, supports_filter=True
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.path: str = cfg.get("path", ":memory:")
        self.dimension: int = int(cfg.get("dim", 256))
        self.kb_id: str = str(cfg.get("kb_id", "default"))
        self._table = f"chunks_{_sanitize(self.kb_id)}"
        self._fts = f"{self._table}_fts"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._ensure_schema()

    # -- connection / schema ------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            self._conn = conn
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self._table} (
                    rowid INTEGER PRIMARY KEY,
                    chunk_id TEXT UNIQUE NOT NULL,
                    doc_id TEXT NOT NULL,
                    kb_id TEXT NOT NULL,
                    atom_ids TEXT NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    page INTEGER,
                    bbox TEXT,
                    metadata TEXT NOT NULL DEFAULT '{{}}',
                    edited INTEGER NOT NULL DEFAULT 0,
                    version INTEGER NOT NULL DEFAULT 1,
                    vector BLOB NOT NULL
                )"""
            )
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self._fts,)
            ).fetchone()
            if not exists:
                conn.execute(
                    f"""CREATE VIRTUAL TABLE {self._fts} USING fts5(
                        text, chunk_id UNINDEXED,
                        content='{self._table}', content_rowid='rowid',
                        tokenize='trigram'
                    )"""
                )
                conn.execute(
                    f"""CREATE TRIGGER {self._table}_ai AFTER INSERT ON {self._table} BEGIN
                        INSERT INTO {self._fts}(rowid, text, chunk_id)
                        VALUES (new.rowid, new.text, new.chunk_id);
                    END"""
                )
                conn.execute(
                    f"""CREATE TRIGGER {self._table}_ad AFTER DELETE ON {self._table} BEGIN
                        INSERT INTO {self._fts}({self._fts}, rowid, text, chunk_id)
                        VALUES ('delete', old.rowid, old.text, old.chunk_id);
                    END"""
                )
                conn.execute(
                    f"""CREATE TRIGGER {self._table}_au AFTER UPDATE OF text ON {self._table} BEGIN
                        INSERT INTO {self._fts}({self._fts}, rowid, text, chunk_id)
                        VALUES ('delete', old.rowid, old.text, old.chunk_id);
                        INSERT INTO {self._fts}(rowid, text, chunk_id)
                        VALUES (new.rowid, new.text, new.chunk_id);
                    END"""
                )
            # NOTE: the connection stays open for the lifetime of the store -
            # ":memory:" databases must not be re-opened, and file databases are
            # faster on a persistent WAL connection.

    # -- filter translation --------------------------------------------------
    @staticmethod
    def _field_sql(field: str, table: str) -> str:
        if field in _REAL_COLUMNS:
            return f"{table}.{field}"
        return f"json_extract({table}.metadata, '$.{field}')"

    def _clause_sql(self, clause: dict[str, Any]) -> tuple[str, list[Any]]:
        """One FilterExpr clause -> (SQL predicate, params)."""
        field = str(clause["field"])
        op = str(clause["op"])
        value = clause.get("value")
        col = self._field_sql(field, self._table)
        if op == "eq":
            return f"CAST({col} AS TEXT) = CAST(? AS TEXT)", [value]
        if op == "ne":
            return f"CAST({col} AS TEXT) <> CAST(? AS TEXT)", [value]
        if op in ("gt", "ge", "lt", "le"):
            symbol = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}[op]
            return f"{col} {symbol} ?", [float(value) if isinstance(value, (int, float)) else value]
        if op == "in":
            values = value if isinstance(value, list) else [value]
            placeholders = ",".join("?" for _ in values)
            return f"CAST({col} AS TEXT) IN ({placeholders})", [str(v) for v in values]
        if op == "contains":
            return f"CAST({col} AS TEXT) LIKE ?", [f"%{str(value)}%"]
        raise PluginContractError(
            "unsupported filter operator", details={"op": op, "field": field}
        )

    def _clause_group(self, clauses: list[dict[str, Any]], join: str) -> tuple[str, list[Any]]:
        pieces: list[str] = []
        params: list[Any] = []
        for clause in clauses:
            sql, params_ = self._clause_sql(clause)
            pieces.append(sql)
            params.extend(params_)
        if not pieces:
            return "", params
        return "(" + f" {join} ".join(pieces) + ")", params

    def _predicate(self, expr: FilterExpr | None) -> tuple[str, list[Any]]:
        """Return the WHERE predicate (without the ``WHERE`` keyword)."""
        if expr is None:
            return "", []
        parts: list[str] = []
        params: list[Any] = []
        for clauses, join in ((expr.and_, "AND"), (expr.or_, "OR")):
            if not clauses:
                continue
            sql, params_ = self._clause_group(clauses, join)
            if sql:
                parts.append(sql)
                params.extend(params_)
        return " AND ".join(parts), params

    def _where(self, expr: FilterExpr | None) -> tuple[str, list[Any]]:
        pred, params = self._predicate(expr)
        return (f"WHERE {pred}" if pred else ""), params

    # -- SPI -----------------------------------------------------------------
    async def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        def _run() -> None:
            with self._lock:
                conn = self._conn
                assert conn is not None
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
                    conn.execute(
                        f"""INSERT INTO {self._table}
                            (chunk_id, doc_id, kb_id, atom_ids, text, token_count,
                             page, bbox, metadata, edited, version, vector)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                            ON CONFLICT(chunk_id) DO UPDATE SET
                                text=excluded.text, token_count=excluded.token_count,
                                page=excluded.page, bbox=excluded.bbox,
                                metadata=excluded.metadata, edited=excluded.edited,
                                version=excluded.version, vector=excluded.vector
                        """,
                        (chunk.chunk_id, chunk.doc_id, chunk.kb_id,
                         json.dumps(chunk.atom_ids, ensure_ascii=False), chunk.text,
                         chunk.token_count, chunk.page,
                         json.dumps(chunk.bbox) if chunk.bbox else None,
                         json.dumps(chunk.metadata, ensure_ascii=False),
                         int(chunk.edited), chunk.version,
                         _vec_to_blob(chunk.vector)),
                    )
        await asyncio.to_thread(_run)

    async def delete(self, chunk_ids: list[str]) -> None:
        def _run() -> None:
            with self._lock:
                conn = self._conn
                assert conn is not None
                for chunk_id in chunk_ids:
                    conn.execute(f"DELETE FROM {self._table} WHERE chunk_id=?", (chunk_id,))
        await asyncio.to_thread(_run)

    async def search_dense(
        self, vector: list[float], *, top_k: int, filter_expr: FilterExpr | None = None
    ) -> list[ScoredChunk]:
        if len(vector) != self.dimension:
            raise PluginContractError(
                "query vector dimension mismatch",
                details={"expected": self.dimension, "got": len(vector)},
            )
        where, params = self._where(filter_expr)
        sql = f"SELECT * FROM {self._table} {where}"
        score = array.array("d", vector)

        def _run() -> list[ScoredChunk]:
            with self._lock:
                conn = self._conn
                assert conn is not None
                out: list[ScoredChunk] = []
                for row in conn.execute(sql, params).fetchall():
                    vec = _blob_to_vec(row["vector"])
                    sim = cosine_similarity(score, vec)
                    out.append(ScoredChunk(chunk=_row_to_chunk(row), score=sim, source="dense"))
                out.sort(key=lambda h: h.score, reverse=True)
                return out[: max(top_k, 0)]
        return await asyncio.to_thread(_run)

    async def search_keyword(
        self, query: str, *, top_k: int, filter_expr: FilterExpr | None = None
    ) -> list[ScoredChunk]:
        if not self.capabilities.supports_bm25:
            return []
        phrase = _quote(query.strip())
        pred, params = self._predicate(filter_expr)
        extra = f" AND {pred}" if pred else ""
        sql = (
            f"SELECT {self._table}.*, bm25({self._fts}) AS rank "
            f"FROM {self._table} "
            f"JOIN {self._fts} ON {self._table}.rowid = {self._fts}.rowid "
            f"WHERE {self._fts} MATCH ?{extra} ORDER BY rank LIMIT ?"
        )
        args: list[Any] = [phrase, *params, max(top_k, 0)]

        def _run() -> list[ScoredChunk]:
            with self._lock:
                conn = self._conn
                assert conn is not None
                out: list[ScoredChunk] = []
                for row in conn.execute(sql, args).fetchall():
                    bm25_score = -float(row["rank"])
                    out.append(ScoredChunk(
                        chunk=_row_to_chunk(row),
                        score=1.0 / (1.0 + max(0.0, bm25_score)),
                        source="bm25",
                    ))
                return out
        return await asyncio.to_thread(_run)

    # -- lifecycle -----------------------------------------------------------
    async def startup(self) -> None:
        self._ensure_schema()

    async def shutdown(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def count(self) -> int:
        def _run() -> int:
            with self._lock:
                conn = self._conn
                assert conn is not None
                return conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]
        return await asyncio.to_thread(_run)
