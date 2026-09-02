"""Task queue (03-ingestion.md §3.7).

The design selects Redis Streams for production; the RX-ING-01 DoD defers that
and uses an **in-process executor** first, keeping the ``TaskQueue`` interface
so a Redis Streams driver can be swapped in later without touching the
pipeline. The interface is deliberately narrow: ``enqueue`` + ``consume``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ragx.core.settings import QueueConfig

logger = logging.getLogger("ragx.ingestion.queue")

#: ``consume(task_id) -> None``; the pipeline's entry point.
Consumer = Callable[[str], Awaitable[None]]


class TaskQueue(Protocol):
    """Queue contract (03-ingestion.md §3.7.2)."""

    async def enqueue(self, task_id: str, *, kb_id: str) -> None: ...

    async def consume(self, consumer: Consumer) -> None: ...


class InProcessQueue:
    """A single-process executor: one asyncio.Queue per kb, a worker loop that
    drains them fairly (round-robin) with bounded concurrency."""

    def __init__(self, cfg: QueueConfig | None = None) -> None:
        self.cfg = cfg or QueueConfig()
        self._queues: dict[str, asyncio.Queue[str]] = {}
        self._consumer: Consumer | None = None
        self._worker: asyncio.Task[Any] | None = None
        self._sem: asyncio.Semaphore | None = None

    async def enqueue(self, task_id: str, *, kb_id: str) -> None:
        q = self._queues.setdefault(kb_id, asyncio.Queue())
        await q.put(task_id)

    async def consume(self, consumer: Consumer) -> None:
        """Start the worker loop (idempotent)."""
        if self._worker is not None:
            return
        self._consumer = consumer
        self._sem = asyncio.Semaphore(self.cfg.worker_concurrency)
        self._worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            # fair round-robin across kb queues (03-ingestion.md §3.7.2)
            kb_ids = list(self._queues)
            if not kb_ids:
                await asyncio.sleep(0.05)
                continue
            processed = False
            for kb_id in kb_ids:
                q = self._queues.get(kb_id)
                if q is None or q.empty():
                    continue
                task_id = q.get_nowait()
                assert self._consumer is not None and self._sem is not None
                async with self._sem:
                    try:
                        await self._consumer(task_id)
                    except Exception:  # noqa: BLE001 - worker must not die
                        logger.exception("task %s failed", task_id)
                q.task_done()
                processed = True
            if not processed:
                # Every queue was empty this pass — yield to the event loop so
                # the worker never spins a no-await busy loop (which would
                # starve the API process that hosts it).
                await asyncio.sleep(0.05)

    async def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
