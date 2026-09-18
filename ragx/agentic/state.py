"""Agentic state schema (07-agentic.md §7.2.1).

TypedDict for the LangGraph state graph, plus the Pydantic models for
individual tasks and verification results.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from ragx.core.models import Citation


class Task(BaseModel):
    """One sub-task in the agentic plan (§7.2.1)."""

    task_id: str                         # task_<ulid>
    seed_query: str                      # sub-query seed
    sub_answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    status: str = "pending"              # pending | running | done | failed
    retries: int = 0                     # seed-query retry count (max 1)
    no_data: bool = False                # [NO DATA] marker


class VerifyResult(BaseModel):
    """Verification outcome (§7.2.1)."""

    passed: bool
    issues: str = ""
    confidence: float = 0.0


class AgenticState(TypedDict, total=False):
    """LangGraph state — the graph's shared mutable state (§7.2.1).

    Fields are optional (``total=False``) so nodes can update only what they
    produce. ``sub_answers`` uses the ``append`` annotation so parallel
    executor results accumulate.
    """

    query: str
    plan: list[Task]
    sub_answers: Annotated[list[str], "append"]
    context: str
    verify_result: VerifyResult
    budget_used: int
    mode_branch: str                     # empty | discovery | answer
    discovery_topics: list[str]          # scope-discovery topic titles
