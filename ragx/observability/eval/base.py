"""Evaluator protocol and models (10-observability.md §10.4.1–§10.4.2).

The ``Evaluator`` protocol is the pluggable interface for evaluation
backends (ragas / deepeval / local). Each backend implements
``evaluate(case, result) -> dict[str, float]`` returning ``{metric_name: score}``
where scores are in [0, 1] (hallucination is inverted: lower is better).

Eval set format (§10.4.1): ``tests/eval/ragx_eval.jsonl``, one JSON object per
line with ``question``, ``ground_truth``, ``kb_id``, optional
``expected_citations``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from ragx.core.models import QueryResult


class EvalCase(BaseModel):
    """One evaluation case (§10.4.1)."""

    question: str
    ground_truth: str
    kb_id: str
    expected_citations: list[str] = Field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating one case across all metrics."""

    question: str
    metrics: dict[str, float] = field(default_factory=dict)
    # Metric -> score; NaN means the backend doesn't support this metric.

    def metric_values(self) -> dict[str, float]:
        """Only finite (non-NaN) metric values."""
        return {k: v for k, v in self.metrics.items() if v == v}  # NaN != NaN


#: The full metric catalogue (§10.4.2).
ALL_METRICS: list[str] = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness",
    "hallucination",
    "citation_accuracy",
]

#: Metrics where higher is better (hallucination is inverted).
HIGHER_IS_BETTER: set[str] = set(ALL_METRICS) - {"hallucination"}


class Evaluator(Protocol):
    """Pluggable evaluation backend (§10.4.2)."""

    name: str

    async def evaluate(self, case: EvalCase, result: QueryResult) -> EvalResult:
        """Evaluate one case. Returns metrics dict; unsupported metrics are NaN."""
        ...
