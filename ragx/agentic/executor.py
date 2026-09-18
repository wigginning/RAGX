"""ParallelExecutor (07-agentic.md §7.5).

Executes sub-tasks in parallel via ``asyncio.gather``. Each task =
Retrieve (reuse HybridRetriever) → Answer (lightweight role) → optional
Seed-Retry (max 1).

Single-task failures are non-fatal: marked ``no_data=True`` with ``[NO DATA]``.
If ALL tasks fail → raise AgenticTasksFailedError(7002) → degrade.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import field
from typing import Any

from pydantic import BaseModel

from ragx.core.exceptions import AgenticTasksFailedError
from ragx.core.models import Citation
from ragx.core.roles import LLMRole
from ragx.spi.interfaces import ChatMessage, ChatRequest

logger = logging.getLogger("ragx.agentic.executor")

#: Max seed-query retries per task (§7.4).
MAX_RETRIES = 1


class SubAnswer(BaseModel):
    """Result of one sub-task execution."""

    answer: str
    citations: list[Citation] = field(default_factory=list)


class RewrittenQuery(BaseModel):
    """Seed-query rewrite output (§7.5 _maybe_seed_retry)."""

    query: str


class ParallelExecutor:
    """Parallel sub-task executor (§7.5)."""

    def __init__(
        self,
        llm: Any,
        retriever: Any,
        assembler: Any,
        embedder: Any,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.assembler = assembler
        self.embedder = embedder
        #: Accumulated token usage across this executor's LLM calls (§7.4).
        self.usage_total = 0

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute all plan tasks in parallel (§7.5)."""
        tasks = state.get("plan", [])
        if not tasks:
            return {**state, "sub_answers": []}

        self.usage_total = 0
        results = await asyncio.gather(
            *(self._exec_task(t, state) for t in tasks),
            return_exceptions=True,
        )

        answers: list[str] = []
        updated_tasks: list[Any] = []

        for task, result in zip(tasks, results, strict=False):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.no_data = True
                answers.append("[NO DATA]")
                logger.warning("task %s failed: %s", task.task_id, result)
            else:
                sub = result
                task.sub_answer = sub.answer
                task.citations = sub.citations
                task.status = "done"
                answers.append(sub.answer)
            updated_tasks.append(task)

        # All tasks failed → degrade (§7.6, 7002)
        if all(t.no_data for t in updated_tasks):
            raise AgenticTasksFailedError(
                "all tasks failed",
                details={"task_count": len(tasks)},
            )

        state["plan"] = updated_tasks
        state["sub_answers"] = answers
        state["budget_used"] = state.get("budget_used", 0) + self.usage_total
        return state

    async def _exec_task(
        self, task: Any, state: dict[str, Any]
    ) -> SubAnswer:
        """Execute one task: Retrieve → Answer (§7.5)."""
        seed_query = task.seed_query
        kb_id = state.get("kb_id", "default")
        filter_expr = state.get("filter_expr")

        qvec = (await self.embedder.embed([seed_query]))[0]
        hits = await self.retriever.retrieve(seed_query, qvec, filter_expr)

        if not hits:
            return await self._maybe_seed_retry(task, state)

        # ContextAssembler expects RetrievalHit, not ScoredChunk.
        # The retriever returns RetrievalHit, so we can use it directly.
        context, citations = self.assembler.assemble(hits)

        resp = await self._answer_llm(seed_query, context, kb_id)
        return SubAnswer(answer=resp.text, citations=citations)

    async def _answer_llm(
        self, query: str, context: str, kb_id: str
    ) -> Any:
        """Lightweight answer generation (role: generate, not synthesize)."""
        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are a RAG assistant. Answer the user's question "
                        "using ONLY the provided context. If the context does not "
                        "contain the answer, say so. Do not invent information."
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=f"Context:\n{context}\n\nQuestion: {query}",
                ),
            ],
            role=LLMRole.GENERATE,
            temperature=0.0,
            kb_id=kb_id,
        )
        resp = await self.llm.chat(req)
        usage: Any = getattr(resp, "usage", None)
        if usage is not None:
            total: Any = getattr(usage, "total", 0) or 0
            if total:
                self.usage_total += int(total)
        return resp

    async def _maybe_seed_retry(
        self, task: Any, state: dict[str, Any]
    ) -> SubAnswer:
        """Seed-query retry for empty recall (max 1 retry, §7.5)."""
        if task.retries >= MAX_RETRIES:
            task.no_data = True
            return SubAnswer(answer="[NO DATA]", citations=[])

        task.retries += 1

        # Rewrite the query
        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "Rewrite the following query to improve retrieval. "
                        "Output JSON: {\"query\": \"...\"}"
                    ),
                ),
                ChatMessage(role="user", content=task.seed_query),
            ],
            role=LLMRole.REWRITE,
            temperature=0.0,
            json_mode=True,
            kb_id=state.get("kb_id", "default"),
        )
        try:
            rewritten = await self.llm.structured(req, schema=RewrittenQuery)
        except Exception:
            task.no_data = True
            return SubAnswer(answer="[NO DATA]", citations=[])

        qvec = (await self.embedder.embed([rewritten.query]))[0]
        hits = await self.retriever.retrieve(
            rewritten.query, qvec, state.get("filter_expr")
        )

        if not hits:
            task.no_data = True
            return SubAnswer(answer="[NO DATA]", citations=[])

        context, citations = self.assembler.assemble(hits)
        resp = await self._answer_llm(rewritten.query, context, state.get("kb_id", "default"))
        return SubAnswer(answer=resp.text, citations=citations)
