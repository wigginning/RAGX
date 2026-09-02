"""AgenticOrchestrator (07-agentic.md §7.6).

Coordinates the full agentic pipeline:
Planner → [Empty|Discovery|Answer] → ParallelExecutor → Synthesis → Verifier → End

Degrade paths (§7.6):
* 7001 Planner failed → Standard fallback
* 7002 All tasks failed → Standard fallback
* 7003 Verify failed → Standard fallback
* 6003 Budget exceeded → force Synthesis (skip remaining tasks)

Produces ``QueryResult(mode="standard", degraded=True)`` on any degrade.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ragx.agentic.executor import ParallelExecutor
from ragx.agentic.planner import Planner
from ragx.agentic.sse import SSEEmitter
from ragx.agentic.synthesis import Verifier, synthesise
from ragx.core.exceptions import (
    AgenticError,
    BudgetExceededError,
)
from ragx.core.models import Citation, QueryResult, TokenUsage
from ragx.core.settings import KBConfig
from ragx.observability.tracing import Tracer
from ragx.retrieval.pipeline import QueryService

logger = logging.getLogger("ragx.agentic.orchestrator")

#: Token budget hard cap (§7.4).
DEFAULT_BUDGET_CAP = 100_000


class AgenticOrchestrator:
    """Full agentic pipeline orchestrator (§7.6)."""

    def __init__(
        self,
        planner: Planner,
        executor: ParallelExecutor,
        verifier: Verifier,
        llm: Any,
        standard_fallback: QueryService,
        kb_cfg: KBConfig,
        trace_id: str = "",
        budget_cap: int = DEFAULT_BUDGET_CAP,
        tracer: Tracer | None = None,
        embedder: Any = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.verifier = verifier
        self.llm = llm
        self.standard_fallback = standard_fallback
        self.kb_cfg = kb_cfg
        self.trace_id = trace_id
        self.budget_cap = budget_cap
        self.tracer = tracer
        self.embedder = embedder
        self.sse = SSEEmitter(trace_id=trace_id)
        self._budget_used = 0

    async def run_agentic(
        self,
        query: str,
        kb_id: str,
        qvec: list[float] | None = None,
    ) -> QueryResult:
        """Run the full agentic pipeline with degradation handling (§7.6)."""
        try:
            return await self._run_graph(query, kb_id, qvec)
        except (AgenticError, BudgetExceededError) as exc:
            code = getattr(exc, "code", None) or 7000
            logger.warning(
                "agentic pipeline degraded (code=%s): %s", code, exc
            )
            std_result = await self._standard_fallback(query, kb_id)
            std_result.degraded = True
            std_result.mode = "standard"
            std_result.details["degrade_reason"] = code
            return std_result
        except Exception as exc:
            logger.error("agentic pipeline unexpected error: %s", exc)
            std_result = await self._standard_fallback(query, kb_id)
            std_result.degraded = True
            std_result.mode = "standard"
            std_result.details["degrade_reason"] = "unexpected"
            return std_result

    async def _run_graph(
        self, query: str, kb_id: str, qvec: list[float] | None
    ) -> QueryResult:
        """Execute the state graph (§7.2.2)."""
        state: dict[str, Any] = {
            "query": query,
            "kb_id": kb_id,
            "budget_used": 0,
        }

        # ── 1. Planner ──────────────────────────────────────────────
        await self.sse.stage_start("planner")
        t0 = time.time()
        state = await self.planner.run(state)
        await self.sse.stage_end("planner", (time.time() - t0) * 1000, state["budget_used"])

        plan = state.get("plan", [])
        branch = state.get("mode_branch", "answer")
        if plan:
            await self.sse.plan(plan, branch)

        # ── 2. Branch: empty ─────────────────────────────────────────
        if branch == "empty" or not plan:
            # Honest answer, no tasks to execute
            return await self._empty_result(query, kb_id, state)

        # ── 3. ParallelExecutor (budget-gated, §7.4) ───────────────────
        await self.sse.stage_start("executor")
        t0 = time.time()
        try:
            self._check_budget(state)
            state = await self.executor.run(state)
        except BudgetExceededError:
            # 6003: skip remaining tasks and force synthesis with whatever
            # sub-answers exist; if none, let the outer handler degrade.
            state.setdefault("sub_answers", [])
            if not any(a for a in state["sub_answers"] if a and a != "[NO DATA]"):
                raise
            logger.warning("agentic budget exceeded (6003); forcing synthesis")
        await self.sse.stage_end("executor", (time.time() - t0) * 1000, state["budget_used"])

        # ── 4. Synthesis ─────────────────────────────────────────────
        await self.sse.stage_start("synthesis")
        t0 = time.time()
        sub_answers = state.get("sub_answers", [])
        answer, cited_ids = await synthesise(
            sub_answers, query, self.llm, kb_id
        )
        # Collect citations from all sub-tasks
        citations: list[Citation] = []
        for task in state.get("plan", []):
            citations.extend(task.citations)
        await self.sse.synthesis(answer, citations)
        await self.sse.stage_end("synthesis", (time.time() - t0) * 1000, state["budget_used"])

        # ── 5. Verifier ──────────────────────────────────────────────
        await self.sse.stage_start("verifier")
        t0 = time.time()
        state = await self.verifier.run({**state, "context": answer})
        await self.sse.stage_end("verifier", (time.time() - t0) * 1000, state["budget_used"])

        verify_result = state.get("verify_result")
        if verify_result is not None and not verify_result.passed:
            raise AgenticError(
                "verification failed",
                details={"issues": verify_result.issues},
                code=7003,
            )

        # ── 6. Build result ──────────────────────────────────────────
        usage = TokenUsage()
        cost_usd = 0.0
        await self.sse.done("agentic", False, usage, cost_usd)

        return QueryResult(
            answer=answer,
            citations=citations,
            mode="agentic",
            trace_id=self.trace_id,
            usage=usage,
            cost_usd=cost_usd,
            degraded=False,
        )

    def _check_budget(self, state: dict[str, Any]) -> None:
        """Raise ``BudgetExceededError(6003)`` once the hard cap is reached (§7.4)."""
        used = state.get("budget_used", 0)
        if used >= self.budget_cap:
            raise BudgetExceededError(
                code=6003,
                message="agentic token budget exceeded",
                details={"used": used, "cap": self.budget_cap},
                trace_id=self.trace_id,
            )

    async def _empty_result(
        self, query: str, kb_id: str, state: dict[str, Any]
    ) -> QueryResult:
        """Return a result for the empty branch (no relevant content)."""
        await self.sse.stage_start("synthesis")
        answer = "Unable to find relevant information in the knowledge base."
        await self.sse.synthesis(answer, [])
        await self.sse.stage_end("synthesis", 0.0, 0)
        await self.sse.done("agentic", False, TokenUsage(), 0.0)
        return QueryResult(
            answer=answer,
            citations=[],
            mode="agentic",
            trace_id=self.trace_id,
            usage=TokenUsage(),
            cost_usd=0.0,
            degraded=False,
            details={"empty": True},
        )

    async def _standard_fallback(
        self, query: str, kb_id: str
    ) -> QueryResult:
        """Fall back to the Standard pipeline (§7.6)."""
        qvec: list[float] = []
        if self.embedder is not None:
            try:
                qvec = (await self.embedder.embed([query]))[0]
            except Exception:
                qvec = []
        return await self.standard_fallback.query(
            query, qvec, kb_id=kb_id, trace_id=self.trace_id
        )
