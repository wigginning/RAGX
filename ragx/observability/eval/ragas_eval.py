"""RAGAS evaluator backend (10-observability.md §10.4.2).

Lazy-imports the ``ragas`` library. If not installed, ``evaluate`` returns
NaN for all metrics except ``citation_accuracy`` (which is self-implemented).
"""

from __future__ import annotations

import logging
import math

from ragx.core.models import QueryResult
from ragx.observability.eval.base import EvalCase, EvalResult
from ragx.observability.eval.citation_accuracy import compute_citation_accuracy

logger = logging.getLogger("ragx.observability.eval.ragas")

#: Metrics supported by RAGAS (the ones RAGAS can compute).
_RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness",
]


class RagasEvaluator:
    """RAGAS-based evaluator (§10.4.2).

    If ``ragas`` is not installed, all ragas-specific metrics return NaN;
    ``citation_accuracy`` always works (self-implemented).
    """

    name = "ragas"

    def __init__(self, llm: str | None = None) -> None:
        self._ragas = None
        self._init_error = None
        if llm:
            self._init_ragas(llm)

    def _init_ragas(self, llm: str) -> None:
        """Lazy-import ragas and configure the LLM judge."""
        try:
            from ragas import evaluate as ragas_evaluate  # type: ignore
            from ragas.metrics import (  # type: ignore
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
            self._ragas = {
                "evaluate": ragas_evaluate,
                "Faithfulness": Faithfulness,
                "AnswerRelevancy": AnswerRelevancy,
                "ContextRecall": ContextRecall,
                "ContextPrecision": ContextPrecision,
            }
        except ImportError as exc:
            self._init_error = str(exc)
            logger.debug("ragas not installed: %s", exc)

    async def evaluate(self, case: EvalCase, result: QueryResult) -> EvalResult:
        metrics: dict[str, float] = {m: math.nan for m in _RAGAS_METRICS}
        metrics["citation_accuracy"] = compute_citation_accuracy(
            result, case.expected_citations
        )

        if self._ragas is None:
            return EvalResult(question=case.question, metrics=metrics)

        # RAGAS requires a specific input format (Dataset).
        # We wrap the single case into a minimal dataset and call evaluate.
        try:
            from ragas import Dataset  # type: ignore

            dataset = Dataset(
                single_turn=[
                    {
                        "question": case.question,
                        "ground_truth": case.ground_truth,
                        "answer": result.answer,
                        "contexts": [c.snippet for c in result.citations]
                        if result.citations else ["(no context)"],
                    }
                ]
            )

            r = self._ragas["evaluate"](
                dataset,
                metrics=[
                    self._ragas["Faithfulness"](),
                    self._ragas["AnswerRelevancy"](),
                    self._ragas["ContextRecall"](),
                    self._ragas["ContextPrecision"](),
                ],
            )
            for metric_name in _RAGAS_METRICS:
                try:
                    metrics[metric_name] = r.metrics[metric_name]
                except (KeyError, TypeError):
                    pass  # leave as NaN
        except Exception as exc:
            logger.warning("ragas evaluation failed: %s", exc)

        return EvalResult(question=case.question, metrics=metrics)
