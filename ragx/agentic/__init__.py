"""Agentic RAG pipeline (07-agentic.md).

LangGraph-style state graph: Planner → [Empty|Discovery|Answer] →
ParallelExecutor → Synthesis → Verifier → End.

Hard constraints (§7.4):
* Max 8 sub-tasks (Planner truncates)
* Max 1 seed-query retry per task
* Max 1 verification pass
* Token budget hard cap (default 100K) → BudgetExceededError(6003) → degrade

Degrade paths (§7.6):
* 7001 Planner failed → Standard
* 7002 All tasks failed → Standard
* 7003 Verify failed → Standard
* 6003 Budget exceeded → force Synthesis (skip remaining tasks)
"""

from ragx.agentic.executor import ParallelExecutor
from ragx.agentic.orchestrator import AgenticOrchestrator
from ragx.agentic.planner import Planner
from ragx.agentic.sse import SSEEmitter
from ragx.agentic.state import AgenticState, Task, VerifyResult
from ragx.agentic.synthesis import Verifier, synthesise

__all__ = [
    "AgenticOrchestrator",
    "AgenticState",
    "ParallelExecutor",
    "Planner",
    "SSEEmitter",
    "Task",
    "VerifyResult",
    "synthesise",
]
