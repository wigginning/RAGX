"""SSE Stage Events (07-agentic.md §7.7).

Emits server-sent events for the agentic pipeline stages. Events are
queued in a ``asyncio.Queue`` and consumed by the SSE stream handler.

Event types: stage_start, stage_end, plan, task_update, synthesis, done.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any


class SSEEmitter:
    """Emits agentic pipeline events to an async queue for SSE streaming."""

    def __init__(self, trace_id: str = "") -> None:
        self.trace_id = trace_id
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit a single event (§7.7)."""
        evt = {
            "type": event_type,
            "trace_id": self.trace_id,
            "ts": datetime.now(UTC).isoformat(),
            **payload,
        }
        await self._queue.put(json.dumps(evt, ensure_ascii=False))

    async def get(self) -> str:
        """Get the next event from the queue (blocks until one is available)."""
        return await self._queue.get()

    @property
    def queue(self) -> asyncio.Queue[str]:
        return self._queue

    def is_empty(self) -> bool:
        return self._queue.empty()

    # -- convenience emitters --------------------------------------------
    async def stage_start(self, stage: str) -> None:
        await self.emit("stage_start", {"stage": stage})

    async def stage_end(
        self, stage: str, duration_ms: float = 0.0, budget_used: int = 0
    ) -> None:
        await self.emit("stage_end", {
            "stage": stage,
            "duration_ms": duration_ms,
            "budget_used": budget_used,
        })

    async def plan(self, tasks: list[Any], branch: str) -> None:
        await self.emit("plan", {
            "tasks": [
                {"task_id": t.task_id, "seed_query": t.seed_query}
                for t in tasks
            ],
            "branch": branch,
        })

    async def task_update(
        self,
        task_id: str,
        status: str,
        sub_answer: str | None = None,
        citations: list[Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"task_id": task_id, "status": status}
        if sub_answer is not None:
            payload["sub_answer"] = sub_answer
        if citations is not None:
            payload["citations"] = [
                c.model_dump() for c in citations
            ]
        await self.emit("task_update", payload)

    async def synthesis(self, draft: str, citations: list[Any]) -> None:
        await self.emit("synthesis", {
            "draft": draft,
            "citations": [c.model_dump() for c in citations],
        })

    async def done(
        self,
        mode: str,
        degraded: bool,
        usage: Any,
        cost_usd: float,
    ) -> None:
        await self.emit("done", {
            "mode": mode,
            "degraded": degraded,
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else usage,
            "cost_usd": cost_usd,
            "trace_id": self.trace_id,
        })
