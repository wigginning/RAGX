"""Agentic Planner (07-agentic.md §7.3).

Two-stage planning:
1. **Intent decision** — lightweight LLM call (role: plan) → empty|discovery|answer.
2. **Answer planning** — produce ≤8 sub-tasks with seed queries.

Optional scope discovery (§7.3.2) when the query is too vague: explore topic
distribution via GraphStore.topics, then re-plan.

Hard constraint: max 8 tasks (Planner truncates, §7.4).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ragx.core.exceptions import AgenticPlanError
from ragx.core.roles import LLMRole
from ragx.core.settings import KBConfig
from ragx.llm.prompts import PromptRegistry
from ragx.spi.interfaces import ChatMessage, ChatRequest

logger = logging.getLogger("ragx.agentic.planner")

#: Hard cap on sub-tasks (§7.4).
MAX_TASKS = 8


class IntentDecision(BaseModel):
    """Planner intent output (§7.3.1)."""

    branch: str = Field(description="empty | discovery | answer")
    reason: str = ""


class TaskPlan(BaseModel):
    """Answer plan output (§7.3.3)."""

    tasks: list[dict[str, str]] = Field(default_factory=list)


class Planner:
    """Two-stage planner: intent decision → answer plan (§7.3)."""

    def __init__(
        self,
        llm: Any,
        kb_cfg: KBConfig,
        prompts: PromptRegistry | None = None,
        graph_store: Any | None = None,
    ) -> None:
        self.llm = llm
        self.kb_cfg = kb_cfg
        self.prompts = prompts or PromptRegistry()
        self.graph_store = graph_store

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the full planning pipeline (§7.3.1)."""
        intent = await self._intent_decision(state["query"])

        if intent.branch == "empty":
            return {**state, "mode_branch": "empty", "plan": []}

        if intent.branch == "discovery":
            state = await self._scope_discovery(state)

        tasks = await self._answer_plan(state)
        if len(tasks) > MAX_TASKS:
            tasks = tasks[:MAX_TASKS]

        state["plan"] = tasks
        state["mode_branch"] = "answer"
        return state

    async def _intent_decision(self, query: str) -> IntentDecision:
        """Lightweight intent classification (role: plan, cheap model)."""
        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are a RAG query intent classifier. "
                        "Given a query, decide if it can be answered directly (answer), "
                        "needs scope discovery (discovery), or the corpus likely has no "
                        "relevant content (empty). "
                        "Output JSON: {\"branch\": \"empty|discovery|answer\", \"reason\": \"...\"}."
                    ),
                ),
                ChatMessage(role="user", content=query),
            ],
            role=LLMRole.PLAN,
            temperature=0.0,
            json_mode=True,
        )
        try:
            return await self.llm.structured(req, schema=IntentDecision)
        except Exception as exc:
            raise AgenticPlanError(
                "intent decision failed",
                details={"error": str(exc)},
            ) from exc

    async def _scope_discovery(self, state: dict[str, Any]) -> dict[str, Any]:
        """Explore topic distribution for vague queries (§7.3.2)."""
        state["query"]
        topics: list[str] = []

        if self.graph_store is not None:
            try:
                # Use a dummy vector for topic discovery (the store handles it).
                topic_results = await self.graph_store.topics(
                    [0.0] * 8, top_k=9
                )
                topics = [t.title for t in topic_results]
            except Exception as exc:
                logger.warning("scope discovery via graph_store failed: %s", exc)

        state["discovery_topics"] = topics
        return state

    async def _answer_plan(self, state: dict[str, Any]) -> list[Any]:
        """Produce sub-tasks with seed queries (§7.3.3)."""
        from ragx.agentic.state import Task

        topics_context = ""
        if state.get("discovery_topics"):
            topics_context = (
                "\nAvailable topics: " + ", ".join(state["discovery_topics"])
            )

        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "You are a RAG sub-query planner. Given a user query, "
                        "produce up to 8 sub-tasks with seed queries that together "
                        "cover the user's intent. "
                        "Output JSON: {\"tasks\": [{\"seed_query\": \"...\"}, ...]}"
                        f"{topics_context}"
                    ),
                ),
                ChatMessage(role="user", content=state["query"]),
            ],
            role=LLMRole.PLAN,
            temperature=0.0,
            json_mode=True,
        )
        try:
            plan = await self.llm.structured(req, schema=TaskPlan)
        except Exception as exc:
            raise AgenticPlanError(
                "answer plan failed",
                details={"error": str(exc)},
            ) from exc

        tasks: list[Task] = []
        for t in plan.tasks[:MAX_TASKS]:
            seed_query = t.get("seed_query", state["query"])
            if seed_query:
                tasks.append(Task(
                    task_id=f"task_{self._ulid()}",
                    seed_query=seed_query,
                ))
        return tasks

    @staticmethod
    def _ulid() -> str:
        from ragx.core.ids import ulid

        return ulid()
