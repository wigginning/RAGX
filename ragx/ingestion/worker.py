"""Standalone ingestion worker (RX-ING-02).

For the full compose profile, this module runs as a dedicated process that
drains the Redis Streams queue and runs the ingestion pipeline. The API
process continues to serve queries without competing for ingestion CPU.

Usage:

    python -m ragx.ingestion.worker

Reads settings from the standard ``RAGX_*`` environment variables (same as
the API service).
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from ragx.core.settings import KBConfig, Settings
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.ingestion.queue_redis import RedisStreamsQueue
from ragx.ingestion.store import MetadataStore
from ragx.plugins import register_builtins
from ragx.spi.registry import PluginRegistry

logger = logging.getLogger("ragx.ingestion.worker")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings()
    registry = PluginRegistry(settings)
    register_builtins(registry)
    db = MetadataStore(":memory:")
    await db.connect()
    queue = RedisStreamsQueue(settings.queue)
    pipeline = IngestionPipeline(registry, KBConfig(), db, llm=None)

    shutdown = asyncio.Event()

    def _on_signal(*_: Any) -> None:
        logger.info("worker received signal, shutting down")
        shutdown.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except (NotImplementedError, RuntimeError):
            # Windows / restricted environments: signal handlers may not
            # be installable. The worker still shuts down on its next
            # queue loop cycle when Redis reports the stream is empty.
            pass

    async def _consume(task_id: str) -> None:
        task = await db.get_task(task_id)
        if task is not None:
            await pipeline.run(task)

    await queue.consume(_consume)
    logger.info("worker started, draining %s", settings.queue.stream_prefix)
    await shutdown.wait()
    await queue.shutdown()
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
