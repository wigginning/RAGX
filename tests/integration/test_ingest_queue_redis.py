"""Integration test: ingestion pipeline driven by Redis Streams queue (RX-ING-02).

Uses ``fakeredis.aioredis`` so the test runs in CI without a real Redis
service. Asserts the full path: ``submit_ingest`` → ``enqueue`` → worker picks
up → ``IngestionPipeline.run`` produces chunks.
"""

from __future__ import annotations

from typing import Any

import fakeredis.aioredis
import pytest

from ragx.core.models import RawDocument
from ragx.core.settings import KBConfig, QueueConfig, Settings
from ragx.ingestion.dedup import submit_ingest
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.ingestion.queue_redis import RedisStreamsQueue
from ragx.ingestion.store import MetadataStore


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> Any:
    server = fakeredis.aioredis.FakeServer()
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)

    def _fake_from_url(*args: Any, **kwargs: Any) -> Any:
        return fake

    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", _fake_from_url)
    return fake


@pytest.fixture
def queue_cfg() -> QueueConfig:
    return QueueConfig(
        driver="redis",
        stream_prefix="ragx:test:ingest",
        consumer_group="ragx-test-workers",
        block_ms=0,
        worker_concurrency=2,
        redis_url="redis://localhost:6379/0",
    )


async def test_ingest_pipeline_consumes_from_redis_stream(
    queue_cfg: QueueConfig,
    fake_redis: Any,
) -> None:
    """Submit -> Redis XADD -> worker pickup -> pipeline.run -> DONE."""
    settings = Settings()
    settings.queue = queue_cfg
    kb_cfg = KBConfig()
    db = MetadataStore(":memory:")
    await db.connect()
    queue = RedisStreamsQueue(queue_cfg)

    # Build a registry stub.
    stub_registry = _StubRegistry()
    pipeline = IngestionPipeline(
        registry=stub_registry,
        kb_cfg=kb_cfg,
        db=db,
    )

    raw = RawDocument(
        kb_id="default",
        filename="hello.txt",
        mimetype="text/plain",
        content=b"hello world " * 20,
    )
    task = await submit_ingest(raw, db, queue)
    await queue.consume(lambda task_id: _run_task(pipeline, db, task_id))

    import asyncio

    deadline = asyncio.get_event_loop().time() + 10.0
    done = False
    while asyncio.get_event_loop().time() < deadline:
        t = await db.get_task(task.task_id)
        if t is not None and t.status.value == "done":
            done = True
            break
        if t is not None and t.status.value == "failed":
            break
        await asyncio.sleep(0.1)

    await queue.shutdown()
    assert done, f"task did not reach DONE; current status={db.get_task.__name__}"

    final = await db.get_task(task.task_id)
    assert final is not None
    assert final.status.value == "done"
    chunks = await db.get_chunks_by_doc(task.doc_id)
    assert len(chunks) > 0

    await db.close()


async def _run_task(pipeline: IngestionPipeline, db: MetadataStore, task_id: str) -> None:
    task = await db.get_task(task_id)
    if task is not None:
        await pipeline.run(task)


class _StubRegistry:
    """A registry that hands out stub interfaces so the pipeline can run."""

    def resolve_from_kb(self, interface: str, kb_cfg: Any, *, kb_id: str) -> Any:
        if interface == "parser":
            return _StubParser()
        if interface == "embedder":
            return _StubEmbedder()
        if interface == "vector_store":
            return _StubVectorStore()
        return _StubEmbedder()


class _StubParser:
    name = "stub"

    async def parse(self, doc: Any, *, options: Any) -> Any:
        from ragx.core.models import Atom, AtomType
        from ragx.spi.interfaces import ParseResult

        text = doc.content.decode("utf-8", errors="ignore") if isinstance(doc.content, bytes) else str(doc.content)
        return ParseResult(
            atoms=[
                Atom(
                    atom_id="atom_1",
                    doc_id=doc.doc_id,
                    type=AtomType.TEXT,
                    text=text,
                    page=1,
                )
            ]
        )


class _StubEmbedder:
    name = "stub"
    dimension = 8
    max_batch_size = 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]


class _StubVectorStore:
    name = "stub"
    capabilities = type("C", (), {"supports_bm25": True, "supports_filter": True})()

    async def upsert(self, chunks: Any) -> None:
        return None

    async def delete(self, chunk_ids: Any) -> None:
        return None

    async def search_dense(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def search_keyword(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None
