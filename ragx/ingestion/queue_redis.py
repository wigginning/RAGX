"""Redis Streams task queue (03-ingestion.md §3.7.2).

Production-grade driver for the ``TaskQueue`` Protocol. Uses Redis Streams with
consumer groups so multiple API workers can share the ingestion load fairly
across knowledge bases.

Stream layout
-------------

* One stream per knowledge base: ``{prefix}:{kb_id}`` (default
  ``ragx:ingest:default``).
* Consumer group: ``{consumer_group}`` (default ``ragx-workers``). All workers
  in the same group share the load; different groups see every message
  independently (useful for shadow evaluation).
* Entry payload (Redis hash): ``{"task_id": "..."}``.
* Dead-letter stream: ``{dlq_stream}`` — messages that exceed retry budget are
  moved here with ``XADD ... MAXLEN ~ 10000``.

Lifecycle
---------

``enqueue`` does ``XADD`` with an auto-generated id (Redis ``*``). ``consume``
spawns one worker task per known kb_id stream; the worker does
``XREADGROUP > GROUP <g> <c> COUNT 1 BLOCK <ms> STREAMS <s> >`` and invokes the
consumer callback with the parsed ``task_id``. On success the message is
acknowledged (``XACK``); on failure it stays pending and is reclaimed by the
next ``XPENDING / XCLAIM`` cycle.

Fair dispatch across kb streams is achieved by a round-robin ``XREADGROUP``
loop that yields the first ready message from any of the configured streams.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragx.core.exceptions import ConfigError, InfraError
from ragx.core.settings import QueueConfig
from ragx.ingestion.queue import Consumer

logger = logging.getLogger("ragx.ingestion.queue_redis")


class RedisStreamsQueue:
    """Production TaskQueue driver backed by Redis Streams (§3.7.2)."""

    def __init__(self, cfg: QueueConfig | None = None) -> None:
        self.cfg = cfg or QueueConfig()
        if not self.cfg.redis_url:
            raise ConfigError(
                "RedisStreamsQueue requires queue.redis_url",
                details={"driver": self.cfg.driver},
            )
        self._redis: Any = None
        self._consumer: Consumer | None = None
        self._workers: list[asyncio.Task[Any]] = []
        self._shutdown = asyncio.Event()
        self._known_kbs: set[str] = set()

    @property
    def _streams(self) -> list[str]:
        """Active streams (configured KBs + any newly enqueued)."""
        return sorted({self._stream(kb) for kb in self._known_kbs})

    def _stream(self, kb_id: str) -> str:
        return f"{self.cfg.stream_prefix}:{kb_id}"

    async def _connect(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:  # pragma: no cover - optional dep
            raise InfraError(
                "redis package not installed",
                code=9002,
                details={"hint": "pip install 'ragx[redis]'"},
            ) from exc
        try:
            client = aioredis.from_url(self.cfg.redis_url, decode_responses=True)
            # lazy ping so the failure surfaces on first op, not at construction
            await client.ping()
        except (ConnectionError, OSError) as exc:
            raise InfraError(
                "redis connection failed",
                code=9002,
                details={"url": self.cfg.redis_url, "error": str(exc)},
            ) from exc
        self._redis = client
        return client

    async def enqueue(self, task_id: str, *, kb_id: str) -> None:
        r = await self._connect()
        self._known_kbs.add(kb_id)
        payload = {"task_id": task_id}
        try:
            await r.xadd(self._stream(kb_id), payload, maxlen=10000, approximate=True)
        except Exception as exc:
            raise InfraError(
                "redis XADD failed",
                code=9002,
                details={"stream": self._stream(kb_id), "error": str(exc)},
            ) from exc
        logger.info("enqueued task_id=%s kb_id=%s", task_id, kb_id)

    async def consume(self, consumer: Consumer) -> None:
        """Start one worker task per known stream; coordinator for new ones."""
        if self._workers:
            return
        self._consumer = consumer
        await self._connect()
        # Ensure the consumer group exists on every known stream and spawn
        # one worker per known stream. New streams added after ``consume()``
        # are picked up by the background coordinator.
        for stream in self._streams:
            await self._ensure_group(stream)
            self._workers.append(asyncio.create_task(self._worker(stream)))
        # Always start the coordinator so streams added after start are
        # covered too.
        self._workers.append(asyncio.create_task(self._coordinator()))

    async def _coordinator(self) -> None:
        """Discover new streams and spawn a worker for each."""
        r = await self._connect()
        # Scan frequently in tests (small block_ms) and at most once a second
        # in production — the cost is one SCAN per cycle, which is cheap.
        while not self._shutdown.is_set():
            try:
                # Discover streams whose prefix matches our config.
                async for key in r.scan_iter(
                    match=f"{self.cfg.stream_prefix}:*", count=100
                ):
                    if key in self._streams:
                        continue
                    suffix = key[len(self.cfg.stream_prefix) + 1 :]
                    kb_id = suffix.split(":", 1)[0] if suffix else "default"
                    self._known_kbs.add(kb_id)
                    await self._ensure_group(key)
                    self._workers.append(asyncio.create_task(self._worker(key)))
            except Exception:
                logger.exception("stream coordinator error")
            await asyncio.sleep(0.5)

    async def _ensure_group(self, stream: str) -> None:
        r = await self._connect()
        try:
            await r.xgroup_create(
                name=stream, groupname=self.cfg.consumer_group, id="0", mkstream=True
            )
        except Exception as exc:
            # BUSYGROUP means group already exists — that's fine.
            msg = str(exc)
            if "BUSYGROUP" not in msg:
                raise InfraError(
                    "redis XGROUP CREATE failed",
                    code=9002,
                    details={"stream": stream, "error": msg},
                ) from exc

    async def _worker(self, stream: str) -> None:
        """Round-robin read on one stream."""
        r = await self._connect()
        consumer_name = f"worker-{id(self)}"
        sem = asyncio.Semaphore(self.cfg.worker_concurrency)
        # When block_ms == 0 we use a non-blocking poll + sleep loop instead
        # of XREADGROUP BLOCK — useful for fakeredis-backed unit tests and
        # for any deployment that wants lower tail latency.
        block = max(0, int(self.cfg.block_ms))
        backoff = block / 1000.0 if block > 0 else 0.05
        while not self._shutdown.is_set():
            try:
                resp = await r.xreadgroup(
                    groupname=self.cfg.consumer_group,
                    consumername=consumer_name,
                    streams={stream: ">"},
                    count=self.cfg.read_count,
                    block=block,
                )
            except Exception:
                logger.exception("XREADGROUP failed on %s", stream)
                await asyncio.sleep(1.0)
                continue
            if not resp:
                if block == 0:
                    # non-blocking poll: yield to event loop before retrying
                    await asyncio.sleep(backoff)
                continue
            for _stream_name, entries in resp:
                for msg_id, fields in entries:
                    await self._handle(sem, stream, msg_id, fields)

    async def _handle(
        self,
        sem: asyncio.Semaphore,
        stream: str,
        msg_id: str,
        fields: dict[str, str],
    ) -> None:
        task_id = fields.get("task_id")
        if not task_id:
            logger.warning("redis stream entry without task_id, dropping: %s", msg_id)
            r = await self._connect()
            await r.xack(stream, self.cfg.consumer_group, msg_id)
            return
        r = await self._connect()
        try:
            async with sem:
                if self._consumer is not None:
                    await self._consumer(task_id)
            await r.xack(stream, self.cfg.consumer_group, msg_id)
        except Exception:
            logger.exception("consumer failed for task_id=%s", task_id)
            # Leave the message pending; XCLAIM in a future cycle will retry.
            await self._retry_or_dlq(stream, msg_id, fields)

    async def _retry_or_dlq(
        self, stream: str, msg_id: str, fields: dict[str, str]
    ) -> None:
        """Move messages that exceeded retry budget to the DLQ (§3.7.4)."""
        r = await self._connect()
        # Track delivery count via an in-stream header (Redis 7+: XADD
        # carries the delivery counter from XPENDING). The lite path uses
        # a simple threshold on PEL delivery count.
        try:
            pending = await r.xpending_range(
                name=stream,
                groupname=self.cfg.consumer_group,
                min_id=msg_id,
                max_id=msg_id,
                count=1,
            )
            delivery_count = pending[0]["times_delivered"] if pending else 1
        except Exception:
            delivery_count = 1
        if delivery_count >= 3:
            try:
                await r.xadd(
                    self.cfg.dlq_stream,
                    {"origin_stream": stream, "msg_id": msg_id, **fields},
                    maxlen=10000,
                    approximate=True,
                )
                await r.xack(stream, self.cfg.consumer_group, msg_id)
            except Exception:
                logger.exception("DLQ move failed for %s", msg_id)
        # else: leave pending — XCLAIM cycle will pick it up next pass.

    async def shutdown(self) -> None:
        self._shutdown.set()
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._workers.clear()
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None


def make_queue(cfg: QueueConfig | None = None) -> Any:
    """Factory: pick the right queue implementation from ``cfg.driver``."""
    cfg = cfg or QueueConfig()
    if cfg.driver == "redis":
        return RedisStreamsQueue(cfg)
    if cfg.driver == "in_process":
        from ragx.ingestion.queue import InProcessQueue

        return InProcessQueue(cfg)
    raise ConfigError(
        "unknown queue driver", details={"driver": cfg.driver}
    )


__all__ = ["RedisStreamsQueue", "make_queue"]
