"""Agentic planner tests (07-agentic.md §7.3).

Covers: intent decision (empty/discovery/answer branches), scope discovery,
answer planning, task cap (max 8), and plan failure (7001).
"""

from __future__ import annotations

import pytest

from ragx.agentic.planner import MAX_TASKS, IntentDecision, Planner, TaskPlan
from ragx.agentic.state import Task
from ragx.core.exceptions import AgenticPlanError
from ragx.core.settings import KBConfig


class _MockLLM:
    """Mock LLM that returns structured results based on call pattern."""

    def __init__(self, intent: IntentDecision | None = None,
                 plan: TaskPlan | None = None) -> None:
        self._intent = intent or IntentDecision(branch="answer", reason="direct")
        self._plan = plan or TaskPlan(
            tasks=[{"seed_query": f"sub-question {i}"} for i in range(3)]
        )
        self.call_count = 0

    async def structured(self, req, *, schema):
        self.call_count += 1
        if schema == IntentDecision:
            return self._intent
        if schema == TaskPlan:
            return self._plan
        raise ValueError(f"unexpected schema: {schema}")


class TestPlanner:
    async def test_answer_branch(self) -> None:
        """Direct answer → produce plan with sub-tasks."""
        llm = _MockLLM()
        planner = Planner(llm, KBConfig())
        state = {"query": "What is RAG?"}
        result = await planner.run(state)
        assert result["mode_branch"] == "answer"
        assert len(result["plan"]) == 3
        assert all(isinstance(t, Task) for t in result["plan"])
        assert result["plan"][0].seed_query == "sub-question 0"
        assert result["plan"][0].task_id.startswith("task_")

    async def test_empty_branch(self) -> None:
        """Empty branch → no plan, mode_branch=empty."""
        llm = _MockLLM(intent=IntentDecision(branch="empty", reason="no data"))
        planner = Planner(llm, KBConfig())
        state = {"query": "quantum entanglement in quantum physics"}
        result = await planner.run(state)
        assert result["mode_branch"] == "empty"
        assert result["plan"] == []

    async def test_discovery_branch(self) -> None:
        """Discovery branch → scope discovery runs, then answer plan."""
        llm = _MockLLM(intent=IntentDecision(branch="discovery", reason="vague"))
        planner = Planner(llm, KBConfig())
        state = {"query": "tell me about this library"}
        result = await planner.run(state)
        assert result["mode_branch"] == "answer"  # discovery → then answer
        assert "discovery_topics" in result
        assert len(result["plan"]) > 0

    async def test_task_cap_8(self) -> None:
        """Hard cap: max 8 tasks (§7.4)."""
        llm = _MockLLM(
            plan=TaskPlan(
                tasks=[{"seed_query": f"q{i}"} for i in range(15)]
            )
        )
        planner = Planner(llm, KBConfig())
        state = {"query": "complex question"}
        result = await planner.run(state)
        assert len(result["plan"]) == MAX_TASKS == 8

    async def test_intent_failure_raises_plan_error(self) -> None:
        """Intent decision failure → AgenticPlanError(7001)."""
        class FailingLLM:
            async def structured(self, req, *, schema):
                raise ValueError("LLM down")

        planner = Planner(FailingLLM(), KBConfig())
        state = {"query": "test"}
        with pytest.raises(AgenticPlanError):
            await planner.run(state)

    async def test_answer_plan_failure_raises(self) -> None:
        """Answer plan failure → AgenticPlanError(7001)."""
        class PartialLLM:
            async def structured(self, req, *, schema):
                if schema == IntentDecision:
                    return IntentDecision(branch="answer", reason="ok")
                raise ValueError("plan failed")

        planner = PartialLLM()
        planner_obj = Planner(planner, KBConfig())
        state = {"query": "test"}
        with pytest.raises(AgenticPlanError):
            await planner_obj.run(state)
