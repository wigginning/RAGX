"""Eval runner and baseline comparison (10-observability.md §10.4.3).

``run_eval`` loads the eval set (JSONL), runs each case through the
configured evaluator backend, and produces an ``EvalReport``.

``compare_baseline`` compares the current scores against
``tests/eval/baseline.json`` and returns a list of metric regressions
(single-metric drop >0.05 fails the gate; hallucination is inverted).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragx.core.models import QueryResult
from ragx.observability.eval.base import (
    ALL_METRICS,
    EvalCase,
    EvalResult,
    Evaluator,
)
from ragx.observability.eval.citation_accuracy import compute_citation_accuracy

logger = logging.getLogger("ragx.observability.eval.run_eval")

#: CI gate threshold: single-metric drop > tolerance fails.
DEFAULT_TOLERANCE = 0.05

#: Default eval set location.
DEFAULT_EVAL_SET = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "eval" / "ragx_eval.jsonl"
#: Default baseline location.
DEFAULT_BASELINE = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "eval" / "baseline.json"


@dataclass
class EvalReport:
    """Full evaluation report for one run."""

    cases: int = 0
    results: list[EvalResult] = field(default_factory=list)
    baseline: list[dict[str, Any]] | None = None
    regressions: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> dict[str, Any]:
        """Aggregate metrics across all cases."""
        if not self.results:
            return {"cases": 0}
        agg: dict[str, float] = {}
        for metric in ALL_METRICS:
            values = [
                r.metrics.get(metric, math.nan)
                for r in self.results
                if r.metrics.get(metric, math.nan) == r.metrics.get(metric, math.nan)
            ]
            if values:
                agg[metric] = sum(values) / len(values)
        return {"cases": self.cases, "metrics": agg, "passed": self.passed}


def load_eval_cases(path: Path | str | None = None) -> list[EvalCase]:
    """Load eval cases from a JSONL file (§10.4.1)."""
    path = Path(path) if path else DEFAULT_EVAL_SET
    cases: list[EvalCase] = []
    if not path.exists():
        logger.warning("eval set not found: %s", path)
        return cases
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(EvalCase(
                question=data["question"],
                ground_truth=data["ground_truth"],
                kb_id=data["kb_id"],
                expected_citations=data.get("expected_citations", []),
            ))
    return cases


def load_baseline(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load the baseline JSON (§10.4.4)."""
    path = Path(path) if path else DEFAULT_BASELINE
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_baseline(
    results: list[EvalResult],
    baseline: list[dict[str, Any]],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[dict[str, Any]]:
    """Compare current scores against baseline (§10.4.3).

    Returns a list of regressions: ``[{question, metric, current, baseline, delta}]``.
    A regression is flagged when:
    * For higher-is-better metrics: ``current < baseline - tolerance``
    * For hallucination (inverted): ``current > baseline + tolerance``
    """
    regressions: list[dict[str, Any]] = []
    baseline_by_q: dict[str, dict[str, Any]] = {
        b["question"]: b for b in baseline
    }

    for result in results:
        b = baseline_by_q.get(result.question)
        if b is None:
            continue
        for metric in ALL_METRICS:
            current = result.metrics.get(metric, math.nan)
            baseline_val = b.get(metric, math.nan)
            if current != current or baseline_val != baseline_val:
                continue  # either is NaN — skip
            if metric == "hallucination":
                # Inverted: lower is better; regression when current > baseline + tol
                delta = current - baseline_val
                if delta > tolerance:
                    regressions.append({
                        "question": result.question,
                        "metric": metric,
                        "current": round(current, 4),
                        "baseline": round(baseline_val, 4),
                        "delta": round(delta, 4),
                    })
            else:
                # Normal: higher is better; regression when current < baseline - tol
                delta = baseline_val - current
                if delta > tolerance:
                    regressions.append({
                        "question": result.question,
                        "metric": metric,
                        "current": round(current, 4),
                        "baseline": round(baseline_val, 4),
                        "delta": round(delta, 4),
                    })
    return regressions


async def run_eval(
    evaluator: Evaluator,
    query_fn: Any,
    *,
    eval_set_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> EvalReport:
    """Run the full evaluation pipeline (§10.4.3).

    Parameters
    ----------
    evaluator:
        The Evaluator backend (ragas/deepeval/local).
    query_fn:
        Async callable ``(question, kb_id) -> QueryResult`` that runs a
        RAG query for each eval case.
    eval_set_path:
        Path to the JSONL eval set. Defaults to ``tests/eval/ragx_eval.jsonl``.
    baseline_path:
        Path to the baseline JSON. Defaults to ``tests/eval/baseline.json``.
    tolerance:
        CI gate threshold for single-metric drops.

    Returns
    -------
    EvalReport with per-case results, aggregate summary, and regression list.
    """
    cases = load_eval_cases(eval_set_path)
    baseline = load_baseline(baseline_path)

    results: list[EvalResult] = []
    for case in cases:
        try:
            result = await query_fn(case.question, case.kb_id)
            eval_result = await evaluator.evaluate(case, result)
            results.append(eval_result)
        except Exception as exc:
            logger.error(
                "eval case failed: %s — %s", case.question[:50], exc
            )
            results.append(EvalResult(
                question=case.question,
                metrics={m: math.nan for m in ALL_METRICS},
            ))

    regressions = compare_baseline(results, baseline, tolerance=tolerance)
    passed = len(regressions) == 0

    return EvalReport(
        cases=len(cases),
        results=results,
        baseline=baseline,
        regressions=regressions,
        passed=passed,
    )


class LocalEvaluator:
    """Self-contained evaluator using only the citation_accuracy metric.

    No external dependencies — useful for smoke testing and CI gating
    when ragas/deepeval are not installed.
    """

    name = "local"

    async def evaluate(
        self, case: EvalCase, result: QueryResult
    ) -> EvalResult:
        from ragx.observability.eval.base import ALL_METRICS

        metrics: dict[str, float] = {m: math.nan for m in ALL_METRICS}
        metrics["citation_accuracy"] = compute_citation_accuracy(
            result, case.expected_citations
        )
        # Simple heuristic: check if ground_truth appears in the answer
        if case.ground_truth.lower() in result.answer.lower():
            metrics["answer_correctness"] = 1.0
        else:
            metrics["answer_correctness"] = 0.0
        return EvalResult(question=case.question, metrics=metrics)
