"""Ingestion sync-path integration tests (RX-ING-01 DoD).

Covers: duplicate-upload dedup (2004 soft), and crash-safe replay (kill
mid-way -> replay produces no duplicate rows thanks to per-stage checkpoints).
"""

from __future__ import annotations

import pytest

from ragx.core.exceptions import DuplicateDocumentError
from ragx.core.hashing import compute_doc_hash
from ragx.core.models import RawDocument, TaskStatus
from ragx.core.settings import KBConfig
from ragx.ingestion.dedup import submit_ingest
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.ingestion.queue import InProcessQueue
from ragx.ingestion.store import MetadataStore
from ragx.plugins import register_builtins
from ragx.spi.registry import PluginRegistry

SAMPLE = """# 第一章

RAGX 是一个分层插件化的检索增强生成平台。

## 1.1 子标题

它通过向量检索与知识图谱提升问答质量。

| 名称 | 值 |
| --- | --- |
| a | 1 |
"""


@pytest.fixture
async def env():
    db = MetadataStore(":memory:")
    await db.connect()
    reg = PluginRegistry()
    register_builtins(reg)
    kb_cfg = KBConfig()
    queue = InProcessQueue()
    yield db, reg, kb_cfg, queue
    await db.close()


def _raw(kb_id: str = "kb_t", content: str = SAMPLE) -> RawDocument:
    return RawDocument(kb_id=kb_id, filename="sample.md", mimetype="text/markdown", content=content)


async def test_duplicate_upload_dedup(env) -> None:
    db, reg, kb_cfg, queue = env
    first = await submit_ingest(_raw(), db, queue)
    assert first.status == TaskStatus.PENDING
    # same kb + same content -> 2004 soft
    with pytest.raises(DuplicateDocumentError) as ei:
        await submit_ingest(_raw(), db, queue)
    assert ei.value.code == 2004
    assert ei.value.details["existing_doc_id"] == first.doc_id
    # different kb + same content -> allowed
    other = await submit_ingest(_raw(kb_id="kb_other"), db, queue)
    assert other.doc_id != first.doc_id


async def test_pipeline_runs_to_done(env) -> None:
    db, reg, kb_cfg, queue = env
    task = await submit_ingest(_raw(), db, queue)
    pipeline = IngestionPipeline(reg, kb_cfg, db)
    done = await pipeline.run(task)
    assert done.status == TaskStatus.DONE
    assert done.progress == 1.0
    chunks = await db.get_chunks_by_doc(task.doc_id)
    assert chunks, "pipeline must produce chunks"
    # vector store has the chunks
    store = reg.resolve_from_kb("vector_store", kb_cfg, kb_id=task.kb_id)
    assert await store.count() == len(chunks)


async def test_replay_produces_no_duplicates(env) -> None:
    """Kill mid-way (simulate by running only part), then replay the full
    pipeline: completed stages are skipped, no duplicate rows appear."""
    db, reg, kb_cfg, queue = env
    task = await submit_ingest(_raw(), db, queue)
    pipeline = IngestionPipeline(reg, kb_cfg, db)

    # simulate a crash after parsing: run only the parse stage
    atoms = await pipeline._parse(task)
    assert atoms

    # replay the full pipeline from the checkpoint
    done = await pipeline.run(task)
    assert done.status == TaskStatus.DONE

    # no duplicate atoms / chunks
    atoms_after = await db.get_atoms(task.doc_id)
    assert len(atoms_after) == len({a.atom_id for a in atoms_after})
    chunks_after = await db.get_chunks_by_doc(task.doc_id)
    assert len(chunks_after) == len({c.chunk_id for c in chunks_after})

    # vector store has exactly the chunk count (no duplicates from replay)
    store = reg.resolve_from_kb("vector_store", kb_cfg, kb_id=task.kb_id)
    assert await store.count() == len(chunks_after)


async def test_doc_hash_is_stable(env) -> None:
    db, reg, kb_cfg, queue = env
    a = compute_doc_hash(SAMPLE.encode("utf-8"))
    b = compute_doc_hash(SAMPLE.encode("utf-8"))
    assert a == b
    assert len(a) == 64  # sha256 hex
