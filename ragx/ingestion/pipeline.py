"""Ingestion pipeline orchestrator (03-ingestion.md §3.2 / §3.8).

Each stage runs through :func:`run_stage`, which implements the crash-safe
checkpoint wrapper: if a checkpoint already exists for the stage, the stage is
skipped (idempotent replay); otherwise it executes, persists its produced IDs,
and only then advances. Retryable failures back off and re-enter the same stage;
non-retryable failures mark the task ``failed``.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ragx.core.exceptions import (
    ConfigError,
    IngestError,
    LLMError,
    PluginContractError,
    RAGXError,
    UnsupportedFormatError,
)
from ragx.core.models import (
    Atom,
    AtomDescription,
    Chunk,
    EmbeddedChunk,
    IngestTask,
    TaskStatus,
    utcnow,
)
from ragx.core.settings import KBConfig, ParseOptions
from ragx.ingestion.costs import process_atoms
from ragx.ingestion.store import MetadataStore
from ragx.spi.interfaces import DescribeOptions
from ragx.spi.registry import PluginRegistry

logger = logging.getLogger("ragx.ingestion.pipeline")


class RetryPolicy(BaseModel):
    """03-ingestion.md §3.2.3."""

    max_attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.2

    def backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return delay * (1 + random.uniform(-self.jitter, self.jitter))


@dataclass
class StageCheckpoint:
    status: TaskStatus
    produced_ids: list[str]
    started_at: str


#: Non-retryable exceptions -> mark failed immediately (03-ingestion.md §3.2.3).
_NON_RETRYABLE = (UnsupportedFormatError, ConfigError, PluginContractError)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _NON_RETRYABLE):
        return False
    if isinstance(exc, LLMError):
        # non-circuit LLM failures are retryable; circuit-open is not
        return exc.code != 6002
    return isinstance(exc, (IngestError, RAGXError))


def _observe_ingest_stage(stage: TaskStatus, seconds: float) -> None:
    """OBS-02: observe one ingest stage duration (10-observability.md §10.2)."""
    try:
        from ragx.observability.metrics import get_metrics

        get_metrics().ingest_duration.labels(
            stage=str(getattr(stage, "value", stage))
        ).observe(seconds)
    except Exception:  # pragma: no cover - metrics must never break ingestion
        logger.debug("failed to record ingest stage duration", exc_info=True)


def _record_ingest_task(kb_id: str, status: str) -> None:
    """OBS-02: count one ingest task outcome (10-observability.md §10.2)."""
    try:
        from ragx.observability.metrics import get_metrics

        get_metrics().ingest_task_total.labels(kb=kb_id, status=status).inc()
    except Exception:  # pragma: no cover - metrics must never break ingestion
        logger.debug("failed to record ingest task metric", exc_info=True)


async def run_stage(
    task: IngestTask,
    stage: TaskStatus,
    stage_fn: Callable[[IngestTask], Awaitable[list[str]]],
    db: MetadataStore,
    retry: RetryPolicy,
) -> list[str]:
    """Checkpoint wrapper (03-ingestion.md §3.2.4)."""
    ckpt = await db.get_checkpoint(task.task_id, stage)
    if ckpt:
        return ckpt  # already completed -> skip

    task.status = stage
    task.updated_at = utcnow()
    await db.save_task(task)

    started = time.perf_counter()
    try:
        produced = await stage_fn(task)
    except Exception as e:  # noqa: BLE001 - classify below
        _observe_ingest_stage(stage, time.perf_counter() - started)
        task.attempts += 1
        task.error = f"{getattr(e, 'code', 0)}: {e}"
        if task.attempts >= retry.max_attempts or not _is_retryable(e):
            task.status = TaskStatus.FAILED
        await db.save_task(task)
        if task.status != TaskStatus.FAILED:
            await asyncio.sleep(retry.backoff(task.attempts))
            return await run_stage(task, stage, stage_fn, db, retry)
        raise

    _observe_ingest_stage(stage, time.perf_counter() - started)
    await db.save_checkpoint(task.task_id, stage, produced)
    return produced


class IngestionPipeline:
    """Orchestrates the ingest stages for one task."""

    def __init__(
        self,
        registry: PluginRegistry,
        kb_cfg: KBConfig,
        db: MetadataStore,
        *,
        retry: RetryPolicy | None = None,
        desc_cache: Any = None,
        llm: Any = None,
        prompts: Any = None,
    ) -> None:
        self.registry = registry
        self.kb_cfg = kb_cfg
        self.db = db
        self.retry = retry or RetryPolicy()
        self.desc_cache = desc_cache
        self.llm = llm
        self.prompts = prompts

    async def run(self, task: IngestTask) -> IngestTask:
        try:
            atoms = await self._parse(task)
            descriptions = await self._process(task, atoms)
            chunks = await self._chunk(task, atoms, descriptions)
            await self._embed(task, chunks)
            if self.kb_cfg.flags.kg_enabled:
                await self._build_kg(task, chunks)
            task.status = TaskStatus.DONE
            task.progress = 1.0
            task.updated_at = utcnow()
            await self.db.save_task(task)
        except Exception:
            _record_ingest_task(task.kb_id, "failed")
            raise
        _record_ingest_task(task.kb_id, "done")
        return task

    # -- stages -------------------------------------------------------------
    async def _parse(self, task: IngestTask) -> list[Atom]:
        await run_stage(
            task, TaskStatus.PARSING, self._do_parse, self.db, self.retry
        )
        return await self.db.get_atoms(task.doc_id)

    async def _do_parse(self, task: IngestTask) -> list[str]:
        parser = self.registry.resolve_from_kb("parser", self.kb_cfg, kb_id=task.kb_id)
        raw = await self.db.get_doc(task.doc_id)
        if raw is None:
            raise IngestError("document not found", details={"doc_id": task.doc_id})
        result = await parser.parse(raw, options=ParseOptions())
        await self.db.save_atoms(result.atoms)
        return [a.atom_id for a in result.atoms]

    async def _process(self, task: IngestTask, atoms: list[Atom]) -> list[AtomDescription]:
        if not self.kb_cfg.flags.vlm_enabled:
            return []
        produced = await run_stage(
            task, TaskStatus.PROCESSING,
            lambda t: self._do_process(t, atoms), self.db, self.retry,
        )
        return produced  # descriptions are not persisted in the sync path

    async def _do_process(self, task: IngestTask, atoms: list[Atom]) -> list[str]:
        small = self.registry.resolve_from_kb("processor", self.kb_cfg, kb_id=task.kb_id) \
            if self.kb_cfg.processor else None
        if small is None:
            return []
        descs = await process_atoms(
            atoms,
            vlm_enabled=True,
            small_proc=small,
            large_proc=small,
            cache=self.desc_cache,
            opts=DescribeOptions(),
        )
        return [d.atom_id for d in descs]

    async def _chunk(
        self, task: IngestTask, atoms: list[Atom], descriptions: list[AtomDescription]
    ) -> list[Chunk]:
        await run_stage(
            task, TaskStatus.CHUNKING,
            lambda t: self._do_chunk(t, atoms, descriptions), self.db, self.retry,
        )
        return await self.db.get_chunks_by_doc(task.doc_id)

    async def _do_chunk(
        self, task: IngestTask, atoms: list[Atom], descriptions: list[AtomDescription]
    ) -> list[str]:
        from ragx.chunking import Chunker

        desc_map = {d.atom_id: d for d in descriptions}
        embedder = self.registry.resolve_from_kb("embedder", self.kb_cfg, kb_id=task.kb_id)
        chunker = Chunker(self.kb_cfg.chunking, embedder=embedder)
        chunks = await chunker.chunk(
            atoms, kb_id=task.kb_id, doc_id=task.doc_id, descriptions=desc_map
        )
        await self.db.save_chunks(chunks)
        return [c.chunk_id for c in chunks]

    async def _embed(self, task: IngestTask, chunks: list[Chunk]) -> None:
        await run_stage(
            task, TaskStatus.EMBEDDING,
            lambda t: self._do_embed(t, chunks), self.db, self.retry,
        )

    async def _do_embed(self, task: IngestTask, chunks: list[Chunk]) -> list[str]:
        embedder = self.registry.resolve_from_kb("embedder", self.kb_cfg, kb_id=task.kb_id)
        store = self.registry.resolve_from_kb("vector_store", self.kb_cfg, kb_id=task.kb_id)
        texts = [c.text for c in chunks]
        vectors = await embedder.embed(texts)
        embedded = [
            EmbeddedChunk(**c.model_dump(), vector=v) for c, v in zip(chunks, vectors, strict=True)
        ]
        await store.upsert(embedded)
        return [c.chunk_id for c in chunks]

    async def _build_kg(self, task: IngestTask, chunks: list[Chunk]) -> None:
        await run_stage(
            task, TaskStatus.KG_BUILDING,
            lambda t: self._do_build_kg(t, chunks), self.db, self.retry,
        )

    async def _do_build_kg(self, task: IngestTask, chunks: list[Chunk]) -> list[str]:
        """Build the knowledge graph for the chunks (05-kg.md §5.4).

        Resolves the graph_store + embedder from the registry and delegates to
        :class:`~ragx.kg.builder.KGBuilder`. When the LLM router is not wired
        in (sync critical path without an LLM), the stage records a checkpoint
        without building — the state machine still advances (degraded, no-op).
        """
        if self.llm is None:
            logger.warning(
                "kg_building stage skipped: no LLM router wired in (doc=%s)",
                task.doc_id,
            )
            return [c.chunk_id for c in chunks]

        from ragx.kg.builder import KGBuilder

        graph_store = self.registry.resolve_from_kb(
            "graph_store", self.kb_cfg, kb_id=task.kb_id
        )
        embedder = self.registry.resolve_from_kb(
            "embedder", self.kb_cfg, kb_id=task.kb_id
        )
        builder = KGBuilder(
            self.llm, graph_store, embedder, self.kb_cfg, self.prompts
        )
        await builder.build_for_chunks(chunks)
        return [c.chunk_id for c in chunks]
