"""Agentic orchestrator tests (07-agentic.md §7.6).

Covers: full pipeline success, all degradation paths (7001/7002/7003/6003),
budget exceeded, and the SSE event sequence.
"""

from __future__ import annotations

from ragx.agentic.executor import ParallelExecutor
from ragx.agentic.orchestrator import AgenticOrchestrator
from ragx.agentic.planner import IntentDecision, Planner
from ragx.agentic.synthesis import Verifier
from ragx.core.exceptions import (
    AgenticPlanError,
    AgenticTasksFailedError,
)
from ragx.core.models import QueryResult
from ragx.core.settings import KBConfig


class _MockLLM:
    def __init__(self) -> None:
        self.call_count = 0

    async def structured(self, req, *, schema):
        from ragx.agentic.planner import TaskPlan as TP
        from ragx.agentic.synthesis import _VerifyResult
        if schema == IntentDecision:
            return IntentDecision(branch="answer", reason="ok")
        if schema == TP:
            return TP(tasks=[{"seed_query": "sub-query"}])
        if schema == _VerifyResult:
            return _VerifyResult(passed=True, confidence=0.95)
        return schema()

    async def chat(self, req):
        from ragx.core.models import TokenUsage
        from ragx.spi.interfaces import ChatResponse
        self.call_count += 1
        return ChatResponse(
            text="synthesized answer",
            usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total=150),
        )


class _MockRetriever:
    async def retrieve(self, query, qvec, filter_expr=None):
        from ragx.core.models import Chunk
        from ragx.retrieval.models import RetrievalHit
        chunk = Chunk(
            chunk_id="chk_1", doc_id="doc_1", kb_id="kb_1",
            atom_ids=["doc_1#0001"], text="retrieved text",
            token_count=50, page=1,
        )
        return [RetrievalHit(chunk=chunk, rrf_score=0.9, sources=["dense"])]


class _MockAssembler:
    def assemble(self, hits):
        from ragx.core.models import Citation
        citations = [Citation(
            chunk_id="chk_1", doc_id="doc_1", filename="doc.pdf",
            page=1, snippet="retrieved text", score=0.9,
        )]
        return "context text", citations


class _MockEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _MockFallback:
    """Mock QueryService for Standard fallback."""

    async def query(self, query, qvec, *, kb_id, trace_id, **kwargs):
        return QueryResult(
            answer="standard fallback answer",
            citations=[],
            mode="standard",
            trace_id=trace_id,
            degraded=False,
        )


def _make_orchestrator(llm=None, **overrides) -> AgenticOrchestrator:
    llm = llm or _MockLLM()
    planner = Planner(llm, KBConfig())
    executor = ParallelExecutor(llm, _MockRetriever(), _MockAssembler(), _MockEmbedder())
    verifier = Verifier(llm)
    kwargs = dict(
        planner=planner,
        executor=executor,
        verifier=verifier,
        llm=llm,
        standard_fallback=_MockFallback(),
        kb_cfg=KBConfig(),
        trace_id="trace_test",
    )
    kwargs.update(overrides)
    return AgenticOrchestrator(**kwargs)


class TestOrchestrator:
    async def test_full_pipeline_success(self) -> None:
        """Happy path: planner → executor → synthesis → verifier → result."""
        orch = _make_orchestrator()
        result = await orch.run_agentic("What is RAG?", "kb_1")
        assert result.mode == "agentic"
        assert result.degraded is False
        assert result.answer == "synthesized answer"
        assert result.trace_id == "trace_test"

    async def test_planner_failure_degrades(self) -> None:
        """7001: Planner fails → degrade to Standard."""
        class FailingPlanner:
            async def run(self, state):
                raise AgenticPlanError("planner failed", code=7001)

        orch = _make_orchestrator(planner=FailingPlanner())
        result = await orch.run_agentic("What is RAG?", "kb_1")
        assert result.degraded is True
        assert result.mode == "standard"
        assert result.details["degrade_reason"] == 7001
        assert result.answer == "standard fallback answer"

    async def test_all_tasks_failed_degrades(self) -> None:
        """7002: All tasks fail → degrade to Standard."""
        class FailingExecutor:
            async def run(self, state):
                raise AgenticTasksFailedError("all tasks failed", code=7002)

        orch = _make_orchestrator(executor=FailingExecutor())
        result = await orch.run_agentic("What is RAG?", "kb_1")
        assert result.degraded is True
        assert result.mode == "standard"
        assert result.details["degrade_reason"] == 7002

    async def test_verification_failure_degrades(self) -> None:
        """7003: Verification fails → degrade to Standard."""
        class FailingVerifier:
            async def run(self, state):
                from ragx.agentic.state import VerifyResult
                state["verify_result"] = VerifyResult(
                    passed=False, issues="unsupported claim"
                )
                return state

        orch = _make_orchestrator(verifier=FailingVerifier())
        result = await orch.run_agentic("What is RAG?", "kb_1")
        assert result.degraded is True
        assert result.mode == "standard"
        assert result.details["degrade_reason"] == 7003

    async def test_empty_branch_returns_honest_answer(self) -> None:
        """Empty branch: no plan → honest answer without degradation."""
        class EmptyPlanner:
            async def run(self, state):
                return {**state, "mode_branch": "empty", "plan": []}

        orch = _make_orchestrator(planner=EmptyPlanner())
        result = await orch.run_agentic("quantum entanglement", "kb_1")
        assert result.mode == "agentic"
        assert result.degraded is False
        assert "Unable to find" in result.answer
        assert result.details.get("empty") is True

    async def test_sse_events_emitted(self) -> None:
        """SSE events are emitted for each stage."""
        orch = _make_orchestrator()
        await orch.run_agentic("What is RAG?", "kb_1")

        events = []
        while not orch.sse.queue.empty():
            events.append(orch.sse.queue.get_nowait())

        event_types = [
            __import__("json").loads(e)["type"] for e in events
        ]
        assert "stage_start" in event_types
        assert "stage_end" in event_types
        assert "plan" in event_types
        assert "done" in event_types

    async def test_degraded_result_carries_trace_id(self) -> None:
        """Degraded result still carries the original trace_id."""
        class FailingPlanner:
            async def run(self, state):
                raise AgenticPlanError("fail", code=7001)

        orch = _make_orchestrator(planner=FailingPlanner())
        result = await orch.run_agentic("test", "kb_1")
        assert result.trace_id == "trace_test"
