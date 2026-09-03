"""Metadata store (03-ingestion.md §3.2 / §3.8).

A single SQLite database (via aiosqlite) holding the ingestion metadata:
documents, tasks, per-stage checkpoints, atoms, chunks, cost records and
traces. The vector store and graph store are separate plugins; this store is
the *source of truth* for task state and idempotency keys.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import aiosqlite

from ragx.core.models import (
    Atom,
    Chunk,
    IngestTask,
    RawDocument,
    TaskStatus,
    utcnow,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kbs (
    kb_id TEXT PRIMARY KEY,
    config TEXT NOT NULL DEFAULT '{}',
    cache_epoch INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mimetype TEXT NOT NULL,
    doc_hash TEXT NOT NULL,
    source_uri TEXT,
    content BLOB,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    produced_ids TEXT NOT NULL DEFAULT '[]',
    started_at TEXT NOT NULL,
    PRIMARY KEY (task_id, stage)
);
CREATE TABLE IF NOT EXISTS atoms (
    atom_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    type TEXT NOT NULL,
    text TEXT,
    payload_ref TEXT,
    page INTEGER,
    bbox TEXT,
    content_hash TEXT NOT NULL,
    context TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    atom_ids TEXT NOT NULL DEFAULT '[]',
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    page INTEGER,
    bbox TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    edited INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS cost_records (
    trace_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    unit_price TEXT NOT NULL DEFAULT '{}',
    cost_usd REAL NOT NULL DEFAULT 0.0,
    cached INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    kb_acl TEXT NOT NULL DEFAULT '[]',
    tenant_id TEXT NOT NULL DEFAULT 'default',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    key_id TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    status_code INTEGER NOT NULL,
    trace_id TEXT NOT NULL,
    action TEXT NOT NULL,
    resource TEXT NOT NULL,
    details TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audit_log_tenant_ts ON audit_log (tenant_id, ts);
CREATE TABLE IF NOT EXISTS quotas (
    tenant_id TEXT NOT NULL,
    period_start TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    upload_bytes INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, period_start)
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(kb_id, doc_hash);
CREATE INDEX IF NOT EXISTS idx_atoms_doc ON atoms(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_cost_kb ON cost_records(kb_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
"""


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class MetadataStore:
    """Async SQLite metadata store."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("MetadataStore not connected")
        return self._db

    # -- documents ----------------------------------------------------------
    async def save_doc(self, doc: RawDocument) -> None:
        await self._conn().execute(
            """INSERT INTO documents
               (doc_id, kb_id, filename, mimetype, doc_hash, source_uri, content, metadata, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                doc.doc_id, doc.kb_id, doc.filename, doc.mimetype, doc.doc_hash,
                doc.source_uri, doc.content,
                json.dumps(doc.metadata, ensure_ascii=False),
                _iso(utcnow()),
            ),
        )
        await self._conn().commit()

    async def get_doc(self, doc_id: str) -> RawDocument | None:
        cur = await self._conn().execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return RawDocument(
            doc_id=row["doc_id"], kb_id=row["kb_id"], filename=row["filename"],
            mimetype=row["mimetype"], content=row["content"] or b"",
            doc_hash=row["doc_hash"], source_uri=row["source_uri"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    async def find_doc_by_hash(self, kb_id: str, doc_hash: str) -> RawDocument | None:
        cur = await self._conn().execute(
            "SELECT * FROM documents WHERE kb_id=? AND doc_hash=?",
            (kb_id, doc_hash),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return RawDocument(
            doc_id=row["doc_id"], kb_id=row["kb_id"], filename=row["filename"],
            mimetype=row["mimetype"], content=row["content"] or b"",
            doc_hash=row["doc_hash"], source_uri=row["source_uri"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # -- tasks --------------------------------------------------------------
    async def save_task(self, task: IngestTask) -> None:
        await self._conn().execute(
            """INSERT INTO tasks
               (task_id, doc_id, kb_id, status, progress, error, attempts, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(task_id) DO UPDATE SET
                 status=excluded.status, progress=excluded.progress,
                 error=excluded.error, attempts=excluded.attempts,
                 updated_at=excluded.updated_at""",
            (
                task.task_id, task.doc_id, task.kb_id, task.status.value,
                task.progress, task.error, task.attempts,
                _iso(task.created_at), _iso(task.updated_at),
            ),
        )
        await self._conn().commit()

    async def get_task(self, task_id: str) -> IngestTask | None:
        cur = await self._conn().execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return IngestTask(
            task_id=row["task_id"], doc_id=row["doc_id"], kb_id=row["kb_id"],
            status=TaskStatus(row["status"]), progress=row["progress"],
            error=row["error"], attempts=row["attempts"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def get_task_by_doc(self, doc_id: str) -> IngestTask | None:
        cur = await self._conn().execute(
            "SELECT * FROM tasks WHERE doc_id=? ORDER BY created_at DESC LIMIT 1",
            (doc_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return IngestTask(
            task_id=row["task_id"], doc_id=row["doc_id"], kb_id=row["kb_id"],
            status=TaskStatus(row["status"]), progress=row["progress"],
            error=row["error"], attempts=row["attempts"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # -- checkpoints --------------------------------------------------------
    async def save_checkpoint(
        self, task_id: str, stage: TaskStatus, produced_ids: list[str]
    ) -> None:
        await self._conn().execute(
            """INSERT INTO checkpoints (task_id, stage, produced_ids, started_at)
               VALUES (?,?,?,?)
               ON CONFLICT(task_id, stage) DO UPDATE SET
                 produced_ids=excluded.produced_ids, started_at=excluded.started_at""",
            (task_id, stage.value, json.dumps(produced_ids), _iso(utcnow())),
        )
        await self._conn().commit()

    async def get_checkpoint(self, task_id: str, stage: TaskStatus) -> list[str] | None:
        cur = await self._conn().execute(
            "SELECT produced_ids FROM checkpoints WHERE task_id=? AND stage=?",
            (task_id, stage.value),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return json.loads(row["produced_ids"] or "[]")

    # -- atoms --------------------------------------------------------------
    async def save_atoms(self, atoms: list[Atom]) -> None:
        for atom in atoms:
            await self._conn().execute(
                """INSERT INTO atoms
                   (atom_id, doc_id, type, text, payload_ref, page, bbox,
                    content_hash, context, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(atom_id) DO UPDATE SET
                     type=excluded.type, text=excluded.text,
                     payload_ref=excluded.payload_ref, page=excluded.page,
                     bbox=excluded.bbox, content_hash=excluded.content_hash,
                     context=excluded.context, metadata=excluded.metadata""",
                (
                    atom.atom_id, atom.doc_id, atom.type.value, atom.text,
                    atom.payload_ref, atom.page,
                    json.dumps(atom.bbox) if atom.bbox else None,
                    atom.content_hash, atom.context,
                    json.dumps(atom.metadata, ensure_ascii=False),
                ),
            )
        await self._conn().commit()

    async def get_atoms(self, doc_id: str) -> list[Atom]:
        cur = await self._conn().execute(
            "SELECT * FROM atoms WHERE doc_id=? ORDER BY atom_id", (doc_id,)
        )
        rows = await cur.fetchall()
        return [self._row_to_atom(r) for r in rows]

    @staticmethod
    def _row_to_atom(row: aiosqlite.Row) -> Atom:
        from ragx.core.models import AtomType

        return Atom(
            atom_id=row["atom_id"], doc_id=row["doc_id"],
            type=AtomType(row["type"]), text=row["text"],
            payload_ref=row["payload_ref"], page=row["page"],
            bbox=json.loads(row["bbox"]) if row["bbox"] else None,
            content_hash=row["content_hash"], context=row["context"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    # -- chunks -------------------------------------------------------------
    async def save_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            await self._conn().execute(
                """INSERT INTO chunks
                   (chunk_id, doc_id, kb_id, atom_ids, text, token_count,
                    page, bbox, metadata, edited, version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     text=excluded.text, token_count=excluded.token_count,
                     page=excluded.page, bbox=excluded.bbox,
                     metadata=excluded.metadata, edited=excluded.edited,
                     version=excluded.version""",
                (
                    chunk.chunk_id, chunk.doc_id, chunk.kb_id,
                    json.dumps(chunk.atom_ids, ensure_ascii=False), chunk.text,
                    chunk.token_count, chunk.page,
                    json.dumps(chunk.bbox) if chunk.bbox else None,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    int(chunk.edited), chunk.version,
                ),
            )
        await self._conn().commit()

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        cur = await self._conn().execute(
            "SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return Chunk(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"], kb_id=row["kb_id"],
            atom_ids=json.loads(row["atom_ids"] or "[]"), text=row["text"],
            token_count=row["token_count"], page=row["page"],
            bbox=json.loads(row["bbox"]) if row["bbox"] else None,
            metadata=json.loads(row["metadata"] or "{}"),
            edited=bool(row["edited"]), version=row["version"],
        )

    async def get_chunks_by_doc(self, doc_id: str) -> list[Chunk]:
        cur = await self._conn().execute(
            "SELECT * FROM chunks WHERE doc_id=? ORDER BY chunk_id", (doc_id,)
        )
        rows = await cur.fetchall()
        return [
            Chunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], kb_id=r["kb_id"],
                atom_ids=json.loads(r["atom_ids"] or "[]"), text=r["text"],
                token_count=r["token_count"], page=r["page"],
                bbox=json.loads(r["bbox"]) if r["bbox"] else None,
                metadata=json.loads(r["metadata"] or "{}"),
                edited=bool(r["edited"]), version=r["version"],
            )
            for r in rows
        ]

    # -- cost records -------------------------------------------------------
    async def save_cost_record(self, record: Any) -> None:
        await self._conn().execute(
            """INSERT INTO cost_records
               (trace_id, ts, kb_id, tenant_id, role, provider, model,
                prompt_tokens, completion_tokens, unit_price, cost_usd, cached)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.trace_id, record.ts, record.kb_id, record.tenant_id,
                record.role, record.provider, record.model,
                record.prompt_tokens, record.completion_tokens,
                json.dumps(record.unit_price), record.cost_usd, int(record.cached),
            ),
        )
        await self._conn().commit()

    async def sum_daily_cost(self, kb_id: str) -> float:
        """Today's accumulated cost for ``kb_id`` (08-llm.md §8.6.3).

        ``ts`` is stored as UTC ISO-8601 with the ``+00:00`` suffix, so the
        lexicographic ``>=`` comparison against today's UTC midnight is a
        correct date boundary.
        """
        day_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        cur = await self._conn().execute(
            "SELECT COALESCE(SUM(cost_usd),0) AS total FROM cost_records "
            "WHERE kb_id=? AND ts>=?",
            (kb_id, day_start),
        )
        row = await cur.fetchone()
        if row is None:
            return 0.0
        return float(row["total"])

    # -- kbs ----------------------------------------------------------------
    async def save_kb(self, kb_id: str, config: dict[str, Any], cache_epoch: int = 1) -> None:
        await self._conn().execute(
            """INSERT INTO kbs (kb_id, config, cache_epoch, created_at)
               VALUES (?,?,?,?)
               ON CONFLICT(kb_id) DO UPDATE SET
                 config=excluded.config, cache_epoch=excluded.cache_epoch""",
            (kb_id, json.dumps(config, ensure_ascii=False), cache_epoch, _iso(utcnow())),
        )
        await self._conn().commit()

    async def bump_cache_epoch(self, kb_id: str) -> int:
        """O(1) semantic-cache invalidation (08-llm.md §8.5.3)."""
        await self._conn().execute(
            "UPDATE kbs SET cache_epoch = cache_epoch + 1 WHERE kb_id=?", (kb_id,)
        )
        await self._conn().commit()
        cur = await self._conn().execute(
            "SELECT cache_epoch FROM kbs WHERE kb_id=?", (kb_id,)
        )
        row = await cur.fetchone()
        return int(row["cache_epoch"]) if row else 1

    # -- kbs ---------------------------------------------------------------
    async def list_kbs(self) -> list[dict[str, Any]]:
        """List all knowledge bases (for MCP ragx_list_kbs tool, §9.7.1)."""
        cur = await self._conn().execute(
            "SELECT kb_id, config FROM kbs ORDER BY created_at"
        )
        rows = await cur.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            config = json.loads(row["config"]) if row["config"] else {}
            result.append({
                "kb_id": row["kb_id"],
                "name": config.get("name", row["kb_id"]),
            })
        return result

    # -- api keys (09-api.md §9.2.1) ----------------------------------------
    async def save_api_key(
        self,
        *,
        key_id: str,
        key_hash: str,
        key_prefix: str,
        kb_acl: list[str],
        tenant_id: str = "default",
        enabled: bool = True,
    ) -> None:
        await self._conn().execute(
            """INSERT INTO api_keys
               (key_id, key_hash, key_prefix, kb_acl, tenant_id, enabled, created_at)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(key_id) DO UPDATE SET
                 key_hash=excluded.key_hash, key_prefix=excluded.key_prefix,
                 kb_acl=excluded.kb_acl, tenant_id=excluded.tenant_id,
                 enabled=excluded.enabled""",
            (
                key_id, key_hash, key_prefix,
                json.dumps(kb_acl, ensure_ascii=False),
                tenant_id, int(enabled), _iso(utcnow()),
            ),
        )
        await self._conn().commit()

    async def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        cur = await self._conn().execute(
            "SELECT * FROM api_keys WHERE key_hash=?", (key_hash,)
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {
            "key_id": row["key_id"],
            "key_hash": row["key_hash"],
            "key_prefix": row["key_prefix"],
            "kb_acl": json.loads(row["kb_acl"] or "[]"),
            "tenant_id": row["tenant_id"],
            "enabled": bool(row["enabled"]),
        }

    # -- audit log ----------------------------------------------------------
    async def save_audit_entry(
        self,
        audit_id: str,
        ts: str,
        tenant_id: str,
        key_id: str,
        method: str,
        path: str,
        status_code: int,
        trace_id: str,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        await self._conn().execute(
            "INSERT INTO audit_log (audit_id, ts, tenant_id, key_id, method, path, "
            "status_code, trace_id, action, resource, details) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                audit_id,
                ts,
                tenant_id,
                key_id,
                method,
                path,
                status_code,
                trace_id,
                action,
                resource,
                json.dumps(details or {}, ensure_ascii=False),
            ),
        )
        await self._conn().commit()

    async def list_audit_entries(
        self,
        tenant_id: str | None = None,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the most-recent audit entries (optionally filtered by tenant)."""
        if tenant_id is not None:
            cur = await self._conn().execute(
                "SELECT * FROM audit_log WHERE tenant_id=? ORDER BY ts DESC LIMIT ?",
                (tenant_id, limit),
            )
        else:
            cur = await self._conn().execute(
                "SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?", (limit,)
            )
        rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "audit_id": row["audit_id"],
                    "timestamp": row["ts"],
                    "tenant_id": row["tenant_id"],
                    "key_id": row["key_id"],
                    "method": row["method"],
                    "path": row["path"],
                    "status_code": row["status_code"],
                    "trace_id": row["trace_id"],
                    "action": row["action"],
                    "resource": row["resource"],
                    "details": json.loads(row["details"] or "{}"),
                }
            )
        return out

    # -- quotas (per-tenant monthly counters) -------------------------------
    async def bump_quota(
        self, tenant_id: str, field: str, count: int, period_start: str,
    ) -> Any:
        """Increment a tenant's monthly counter atomically.

        ``field`` must be ``"tokens"`` or ``"upload_bytes"`` (the two columns
        in the ``quotas`` table). Returns the post-update :class:`QuotaUsage`.
        """

        if field not in {"tokens", "upload_bytes"}:
            raise ValueError(f"unsupported quota field: {field}")
        await self._conn().execute(
            "INSERT INTO quotas (tenant_id, period_start, tokens, upload_bytes, updated_at) "
            "VALUES (?, ?, 0, 0, ?) "
            "ON CONFLICT(tenant_id, period_start) DO NOTHING",
            (tenant_id, period_start, _iso(utcnow())),
        )
        await self._conn().execute(
            f"UPDATE quotas SET {field} = {field} + ?, updated_at = ? "
            "WHERE tenant_id = ? AND period_start = ?",
            (count, _iso(utcnow()), tenant_id, period_start),
        )
        await self._conn().commit()
        return await self.get_quota(tenant_id, period_start)

    async def get_quota(self, tenant_id: str, period_start: str) -> Any:
        from ragx.api.middleware.quota import QuotaUsage  # local import: avoid cycles

        cur = await self._conn().execute(
            "SELECT * FROM quotas WHERE tenant_id=? AND period_start=?",
            (tenant_id, period_start),
        )
        row = await cur.fetchone()
        if row is None:
            return QuotaUsage(period_start=period_start)
        return QuotaUsage(
            tokens=int(row["tokens"]),
            upload_bytes=int(row["upload_bytes"]),
            period_start=row["period_start"],
        )
