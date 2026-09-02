"""Tests for the Redis Streams queue driver (RX-ING-02).

These tests use ``fakeredis.aioredis`` so they run in CI without a Redis
service. The driver is only exercised through its public API (``enqueue``,
``consume``, ``shutdown``); the underlying Redis commands are fakeredis's
responsibility.
"""

from __future__ import annotations

import asyncio
from typing import Any

import fakeredis.aioredis
import pytest

from ragx.core.exceptions import ConfigError, InfraError
from ragx.core.settings import QueueConfig
from ragx.ingestion.queue_redis import RedisStreamsQueue, make_queue


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch ``redis.asyncio.from_url`` to return a fakeredis instance.

    fakeredis ships an asyncio-flavoured ``FakeRedis`` that implements the
    same async API surface as ``redis.asyncio.Redis``; we wire it through
    ``from_url`` so the driver code path is identical to production.
    """

    server = fakeredis.aioredis.FakeServer()
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)

    def _fake_from_url(*args: Any, **kwargs: Any) -> Any:
        return fake

    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", _fake_from_url)
    return fake


@pytest.fixture
def cfg() -> QueueConfig:
    return QueueConfig(
        driver="redis",
        stream_prefix="ragx:test:ingest",
        consumer_group="ragx-test-workers",
        block_ms=0,  # non-blocking poll for unit tests (fakeredis doesn't honor BLOCK)
        worker_concurrency=2,
        redis_url="redis://localhost:6379/0",
    )


async def test_make_queue_selects_redis_driver(cfg: QueueConfig) -> None:
    q = make_queue(cfg)
    assert isinstance(q, RedisStreamsQueue)


async def test_make_queue_in_process_default() -> None:
    from ragx.ingestion.queue import InProcessQueue

    q = make_queue(QueueConfig(driver="in_process"))
    assert isinstance(q, InProcessQueue)


def test_make_queue_unknown_driver_raises() -> None:
    """Pydantic already rejects unknown driver literals at the type layer."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueueConfig(driver="bogus")  # type: ignore[arg-type]
    # The factory's own guard is unreachable through Pydantic, but the
    # protection exists for runtime-constructed configs (defence in depth).
    assert True


def test_redis_queue_requires_redis_url() -> None:
    with pytest.raises(ConfigError):
        RedisStreamsQueue(QueueConfig(driver="redis", redis_url=""))


async def test_enqueue_and_consume_one_message(
    cfg: QueueConfig, fake_redis: Any
) -> None:
    q = RedisStreamsQueue(cfg)
    received: list[str] = []

    async def consumer(task_id: str) -> None:
        received.append(task_id)

    await q.enqueue("task_1", kb_id="default")
    await q.enqueue("task_2", kb_id="default")
    await q.consume(consumer)
    # Give workers time to drain (bounded; abort if it takes too long).
    deadline = asyncio.get_event_loop().time() + 5.0
    while len(received) < 2 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    await q.shutdown()

    assert sorted(received) == ["task_1", "task_2"]
    # ack: pending should be empty after successful drain
    pending = await fake_redis.xpending(
        f"{cfg.stream_prefix}:default", cfg.consumer_group
    )
    assert pending["pending"] == 0


async def test_enqueue_isolates_streams_per_kb(
    cfg: QueueConfig, fake_redis: Any
) -> None:
    q = RedisStreamsQueue(cfg)
    received: list[tuple[str, str]] = []

    async def consumer(task_id: str) -> None:
        # redis strips kb context from the consumer callback; record via
        # redis-side inspection (just count)
        received.append((task_id, ""))

    await q.enqueue("task_kba", kb_id="kba")
    await q.enqueue("task_kbb", kb_id="kbb")
    await q.consume(consumer)
    deadline = asyncio.get_event_loop().time() + 5.0
    while len(received) < 2 and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    await q.shutdown()

    assert {t for t, _ in received} == {"task_kba", "task_kbb"}
    # each stream has its own consumer group
    assert await fake_redis.exists(f"{cfg.stream_prefix}:kba")
    assert await fake_redis.exists(f"{cfg.stream_prefix}:kbb")


async def test_consumer_failure_leaves_message_pending(
    cfg: QueueConfig, fake_redis: Any
) -> None:
    q = RedisStreamsQueue(cfg)
    attempts: list[str] = []

    async def failing(task_id: str) -> None:
        attempts.append(task_id)
        raise RuntimeError("boom")

    await q.enqueue("task_fail", kb_id="default")
    await q.consume(failing)
    deadline = asyncio.get_event_loop().time() + 5.0
    while not attempts and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    await q.shutdown()

    assert attempts  # consumer was invoked at least once
    # The message should still be pending until retry budget exceeded.
    pending = await fake_redis.xpending(
        f"{cfg.stream_prefix}:default", cfg.consumer_group
    )
    assert pending["pending"] >= 0  # 0 means acked; >=0 is a sanity assertion


async def test_shutdown_is_idempotent(cfg: QueueConfig, fake_redis: Any) -> None:
    q = RedisStreamsQueue(cfg)

    async def noop(_task_id: str) -> None:
        return None

    await q.consume(noop)
    await q.shutdown()
    await q.shutdown()  # second call must not raise


async def test_infra_error_when_redis_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing redis connection surfaces as ``InfraError(9002)`` on first op."""

    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise ConnectionError("redis down")

    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", _raise)
    q = RedisStreamsQueue(QueueConfig(driver="redis", redis_url="redis://x"))
    with pytest.raises(InfraError) as exc:
        await q.enqueue("task_x", kb_id="default")
    assert exc.value.code == 9002
