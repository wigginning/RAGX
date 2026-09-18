"""InProcessQueue worker-loop regression tests.

The auto-consume worker is hosted inside the API event loop (lite compose
``RAGX_QUEUE.AUTO_CONSUME``). A worker loop that spins without yielding after
draining an empty queue would starve the whole process — this test pins the
fix: after each pass with no work, the loop must ``await`` so other coroutines
stay schedulable.
"""

from __future__ import annotations

import asyncio

import pytest

from ragx.ingestion.queue import InProcessQueue


@pytest.mark.asyncio
async def test_worker_yields_when_queue_drains() -> None:
    queue = InProcessQueue()
    consumed: list[str] = []

    async def consumer(task_id: str) -> None:
        consumed.append(task_id)

    await queue.consume(consumer)

    async def wait_for(predicate, timeout: float = 2.0) -> None:
        async def _poll():
            while not predicate():
                await asyncio.sleep(0.01)
            return True

        await asyncio.wait_for(_poll(), timeout)

    await queue.enqueue("t1", kb_id="kb")
    await wait_for(lambda: bool(consumed))
    assert consumed == ["t1"]

    # The queue is now empty but the kb key persists. The worker must still
    # yield to the event loop; otherwise this sleep can never resume (old bug).
    await asyncio.wait_for(asyncio.sleep(0.1), timeout=1.0)

    # And a second enqueue must still be picked up afterwards.
    await queue.enqueue("t2", kb_id="kb")
    await wait_for(lambda: len(consumed) == 2)
    assert consumed == ["t1", "t2"]

    await queue.shutdown()


@pytest.mark.asyncio
async def test_consume_is_idempotent() -> None:
    queue = InProcessQueue()
    seen: list[str] = []

    async def consumer(task_id: str) -> None:
        seen.append(task_id)

    await queue.consume(consumer)
    first_worker = queue._worker
    await queue.consume(consumer)  # must not spawn a second worker
    assert queue._worker is first_worker
    await queue.shutdown()
