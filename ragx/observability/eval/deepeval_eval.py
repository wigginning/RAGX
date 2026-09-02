"""DeepEval evaluator backend (10-observability.md §10.4.2).

Lazy-imports the ``deepeval`` library. If not installed, all deepeval-specific
metrics return NaN; ``citation_accuracy`` always works (self-implemented).

DeepEval provides broader metric coverage (hallucination, G-Eval custom
metrics) and better pytest integration.
"""

from __future__ import annotations

import logging
import math

from ragx.core.models import QueryResult
from ragx.observability.eval.base import EvalCase, EvalResult
from ragx.observability.eval.citation_accuracy import compute_citation_accuracy

logger = logging.getLogger("ragx.observability.eval.deepeval")

#: Metrics supported by DeepEval.
_DEEPEVAL_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "hallucination",
]


class DeepEvalEvaluator:
    """DeepEval-based evaluator (§10.4.2).

    If ``deepeval`` is not installed, all deepeval-specific metrics return NaN;
    ``citation_accuracy`` always works.
    """

    name = "deepeval"

    def __init__(self, llm: str | None = None) -> None:
        self._deepeval = None
        if llm:
            self._init_deepeval(llm)

    def _init_deepeval(self, llm: str) -> None:
        """Lazy-import deepeval and configure the LLM judge."""
        try:
            from deepeval.metrics import (  # type: ignore
                AnswerRelevancyMetric,
                ContextualPrecisionMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase  # type: ignore
            self._deepeval = {
                "LLMTestCase": LLMTestCase,
                "FaithfulnessMetric": FaithfulnessMetric,
                "AnswerRelevancyMetric": AnswerRelevancyMetric,
                "ContextualRecallMetric": ContextualRecallMetric,
                "ContextualPrecisionMetric": ContextualPrecisionMetric,
            }
        except ImportError as exc:
            logger.debug("deepeval not installed: %s", exc)

    async def evaluate(self, case: EvalCase, result: QueryResult) -> EvalResult:
        metrics: dict[str, float] = {m: math.nan for m in _DEEPEVAL_METRICS}
        metrics["citation_accuracy"] = compute_citation_accuracy(
            result, case.expected_citations
        )

        if self._deepeval is None:
            return EvalResult(question=case.question, metrics=metrics)

        try:
            tc = self._deepeval["LLMTestCase"](
                input=case.question,
                actual_output=result.answer,
                expected_output=case.ground_truth,
                retrieval_contexts=[
                    c.snippet for c in result.citations
                ] if result.citations else ["(no context)"],
            )

            metric_results = {
                "faithfulness": self._deepeval["FaithfulnessMetric"](),
                "answer_relevancy": self._deepeval["AnswerRelevancyMetric"](),
                "context_recall": self._deepeval["ContextualRecallMetric"](),
                "context_precision": self._deepeval["ContextualPrecisionMetric"](),
            }
            for name, metric in metric_results.items():
                metric.measure(tc)
                metrics[name] = float(metric.score)
        except Exception as exc:
            logger.warning("deepeval evaluation failed: %s", exc)

        return EvalResult(question=case.question, metrics=metrics)
