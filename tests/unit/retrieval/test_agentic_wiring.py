"""QueryService agentic wiring tests (07-agentic.md §7.9).

Proves that when ``mode == "agentic"`` and an orchestrator is wired in, the
query service delegates to it; when no orchestrator is wired, the path degrades
to standard (critical-path behaviour).
"""

from __future__ import annotations

from typing import Any, cast

from ragx.core.models import QueryResult, RequestOverride, TokenUsage
from ragx.core.settings import KBConfig
from ragx.llm.prompts import PromptRegistry
from ragx.retrieval.models import RetrievalConfig
from ragx.retrieval.pipeline import QueryService
from ragx.retrieval.router import QueryRouter


class _Retriever:
    async def retrieve(self, query, qvec, filter_expr=None):
        return []


class _Assembler:
    def assemble(self, hits):
        return "", []


class _LLM:
    async def chat(self, req):
        from ragx.spi.interfaces import ChatResponse
        return ChatResponse(text="标准答案", usage=TokenUsage())


class _AgenticOrchestrator:
    """Records delegation and returns an agentic result."""

    def __init__(self):
        self.called = False

    async def run_agentic(self, query, kb_id, qvec=None):
        self.called = True
        return QueryResult(
            answer="agentic 答案", citations=[], mode="agentic",
            trace_id="trace_t", usage=TokenUsage(), cost_usd=0.0,
            degraded=False,
        )


def _kb_cfg(agentic_enabled: bool) -> KBConfig:
    cfg = KBConfig()
    cfg.flags.agentic_enabled = agentic_enabled
    return cfg


def _service(orchestrator, kb_cfg) -> QueryService:
    # Duck-typed mocks stand in for the real HybridRetriever/ContextAssembler.
    return QueryService(
        cast(Any, _Retriever()), cast(Any, _Assembler()),
        QueryRouter(), PromptRegistry(), _LLM(),
        kb_cfg=kb_cfg, retrieval_cfg=RetrievalConfig(), agentic=orchestrator,
    )


async def test_agentic_override_delegates_to_orchestrator() -> None:
    orch = _AgenticOrchestrator()
    svc = _service(orch, _kb_cfg(agentic_enabled=True))
    result = await svc.query(
        "复杂问题", [0.1] * 8, kb_id="kb_1", trace_id="trace_t",
        override=RequestOverride(mode="agentic"),
    )
    assert orch.called is True
    assert result.mode == "agentic"
    assert result.answer == "agentic 答案"


async def test_agentic_without_orchestrator_degrades_to_standard() -> None:
    svc = _service(None, _kb_cfg(agentic_enabled=True))
    result = await svc.query(
        "复杂问题", [0.1] * 8, kb_id="kb_1", trace_id="trace_t",
        override=RequestOverride(mode="agentic"),
    )
    # no orchestrator → degrades; empty recall → honest standard answer
    assert result.mode == "standard"
    assert result.citations == []
