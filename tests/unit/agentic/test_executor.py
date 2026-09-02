"""ParallelExecutor tests (07-agentic.md §7.5).

Covers: parallel task execution, single-task failure isolation ([NO DATA]),
all-tasks-failed → 7002, and seed-query retry (max 1).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragx.agentic.executor import MAX_RETRIES, ParallelExecutor
from ragx.agentic.state import Task
from ragx.core.exceptions import AgenticTasksFailedError
from ragx.core.models import Chunk
from ragx.retrieval.models import RetrievalHit


class _LLM:
    """Answers any chat call; structured rewrite returns a fixed query."""

    def __init__(self, *, fail_chat: bool = False) -> None:
        self.fail_chat = fail_chat
        self.chat_calls = 0

    async def chat(self, req):
        self.chat_calls += 1
        if self.fail_chat:
            raise RuntimeError("boom")
        return type("Resp", (), {"text": "子答案", "usage": None})()

    async def structured(self, req, *, schema):
        return SimpleNamespace(query="重写后的查询")


class _Retriever:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty
        self.queries: list[str] = []

    async def retrieve(self, query, qvec, filter_expr=None):
        self.queries.append(query)
        if self.empty:
            return []
        chunk = Chunk(
            chunk_id="chk_1", doc_id="doc_1", kb_id="kb_1",
            atom_ids=["doc_1#0001"], text="retrieved", token_count=10,
        )
        return [RetrievalHit(chunk=chunk, rrf_score=0.9, sources=["dense"])]


class _Assembler:
    def assemble(self, hits):
        return "context", []


class _Embedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _task(task_id: str = "task_1", seed: str = "seed") -> Task:
    return Task(task_id=task_id, seed_query=seed)


def _state(tasks: list[Task]) -> dict:
    return {"plan": tasks, "kb_id": "kb_1", "budget_used": 0}


class TestParallelExecutor:
    async def test_executes_tasks_and_collects_answers(self) -> None:
        ex = ParallelExecutor(_LLM(), _Retriever(), _Assembler(), _Embedder())
        state = _state([_task("task_1", "q1"), _task("task_2", "q2")])

        out = await ex.run(state)

        assert len(out["sub_answers"]) == 2
        assert all(a == "子答案" for a in out["sub_answers"])
        assert all(t.status == "done" for t in out["plan"])
        assert all(not t.no_data for t in out["plan"])

    async def test_single_task_failure_is_isolated(self) -> None:
        """One task fails → [NO DATA], others complete (not an exception)."""
        ex = ParallelExecutor(_LLM(), _Retriever(), _Assembler(), _Embedder())
        state = _state([_task("task_1", "q1"), _task("task_2", "q2")])
        # make only the first task fail by making chat fail once
        _LLM()
        calls = {"n": 0}

        async def chat(req):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return type("Resp", (), {"text": "ok", "usage": None})()

        ex.llm.chat = chat
        out = await ex.run(state)
        answers = out["sub_answers"]
        assert "[NO DATA]" in answers
        assert "ok" in answers
        assert any(t.no_data for t in out["plan"])
        assert any(t.status == "failed" for t in out["plan"])

    async def test_all_tasks_failed_raises_7002(self) -> None:
        ex = ParallelExecutor(
            _LLM(fail_chat=True), _Retriever(), _Assembler(), _Embedder()
        )
        state = _state([_task("task_1", "q1"), _task("task_2", "q2")])
        with pytest.raises(AgenticTasksFailedError) as ei:
            await ex.run(state)
        assert ei.value.code == 7002

    async def test_empty_recall_hits_retry_cap(self) -> None:
        """Empty recall → rewrite retry (max 1) → still empty → all tasks
        fail → AgenticTasksFailedError(7002) (single task, so [NO DATA] = all)."""
        retriever = _Retriever(empty=True)
        ex = ParallelExecutor(_LLM(), retriever, _Assembler(), _Embedder())

        state = _state([_task("task_1", "q1")])
        with pytest.raises(AgenticTasksFailedError) as ei:
            await ex.run(state)
        assert ei.value.code == 7002
        # retries hit the cap
        assert state["plan"][0].retries == MAX_RETRIES
        assert state["plan"][0].no_data is True

    async def test_seed_retry_recovery(self) -> None:
        """First recall empty, rewritten recall succeeds → answer."""
        class _FlipRetriever:
            def __init__(self):
                self.n = 0

            async def retrieve(self, query, qvec, filter_expr=None):
                self.n += 1
                if self.n == 1:
                    return []
                chunk = Chunk(
                    chunk_id="chk_1", doc_id="doc_1", kb_id="kb_1",
                    atom_ids=["doc_1#0001"], text="retrieved", token_count=10,
                )
                return [RetrievalHit(chunk=chunk, rrf_score=0.9, sources=["dense"])]

        ex = ParallelExecutor(_LLM(), _FlipRetriever(), _Assembler(), _Embedder())
        state = _state([_task("task_1", "q1")])
        out = await ex.run(state)
        assert out["sub_answers"] == ["子答案"]
        assert state["plan"][0].retries == 1
        assert state["plan"][0].status == "done"

    async def test_no_plan_returns_empty(self) -> None:
        ex = ParallelExecutor(_LLM(), _Retriever(), _Assembler(), _Embedder())
        out = await ex.run({"plan": [], "kb_id": "kb_1"})
        assert out["sub_answers"] == []
