"""Evaluation framework tests (10-observability.md §10.4).

Covers: EvalCase/eval set loading, Evaluator protocol, citation_accuracy
computation, baseline comparison (normal + inverted hallucination),
and the LocalEvaluator.
"""

from __future__ import annotations

import json
import math
import tempfile

import pytest

from ragx.core.models import Citation, QueryResult
from ragx.observability.eval.base import (
    EvalCase,
    EvalResult,
)
from ragx.observability.eval.citation_accuracy import compute_citation_accuracy
from ragx.observability.eval.run_eval import (
    LocalEvaluator,
    compare_baseline,
    load_baseline,
    load_eval_cases,
    run_eval,
)


def _make_result(
    answer: str = "answer text",
    citation_ids: list[str] | None = None,
) -> QueryResult:
    citations = [
        Citation(
            chunk_id=cid, doc_id="doc_1", filename="doc.pdf",
            page=1, snippet=f"snippet for {cid}", score=0.9,
        )
        for cid in (citation_ids or [])
    ]
    return QueryResult(
        answer=answer,
        citations=citations,
        mode="standard",
        trace_id="trace_1",
    )


class TestCitationAccuracy:
    def test_perfect_match(self) -> None:
        """All citations match expected → score 1.0."""
        result = _make_result(citation_ids=["chk_1", "chk_2"])
        score = compute_citation_accuracy(result, ["chk_1", "chk_2"])
        assert score == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        """Half of citations match → F1 = 2*0.5*0.5 / (0.5+0.5) = 0.5."""
        result = _make_result(citation_ids=["chk_1", "chk_2"])
        score = compute_citation_accuracy(result, ["chk_1", "chk_3"])
        # tp=1, precision=1/2, recall=1/2, f1=0.5
        assert score == pytest.approx(0.5)

    def test_no_match(self) -> None:
        """No citations match → score 0.0."""
        result = _make_result(citation_ids=["chk_1"])
        score = compute_citation_accuracy(result, ["chk_99"])
        assert score == 0.0

    def test_no_actual_citations(self) -> None:
        """No actual citations → score 0.0 (recall is 0)."""
        result = _make_result(citation_ids=[])
        score = compute_citation_accuracy(result, ["chk_1", "chk_2"])
        assert score == 0.0

    def test_no_expected_citations(self) -> None:
        """No expected citations → NaN (metric not applicable)."""
        result = _make_result(citation_ids=["chk_1"])
        score = compute_citation_accuracy(result, [])
        assert math.isnan(score)

    def test_extra_citations_lower_precision(self) -> None:
        """Extra citations reduce precision → lower F1."""
        result = _make_result(citation_ids=["chk_1", "chk_2", "chk_3", "chk_4"])
        score = compute_citation_accuracy(result, ["chk_1"])
        # tp=1, precision=1/4, recall=1/1=1, f1=2*0.25*1/(0.25+1)=0.4
        assert score == pytest.approx(0.4)


class TestEvalSetLoading:
    def test_load_eval_cases(self) -> None:
        cases = load_eval_cases()
        assert len(cases) == 3
        assert cases[0].question.startswith("LightRAG")
        assert cases[0].expected_citations == ["chk_001", "chk_002"]
        assert cases[0].kb_id == "kb_test"

    def test_load_baseline(self) -> None:
        baseline = load_baseline()
        assert len(baseline) == 3
        assert "question" in baseline[0]
        assert "faithfulness" in baseline[0]

    def test_load_missing_eval_set(self) -> None:
        cases = load_eval_cases("/nonexistent/path.jsonl")
        assert cases == []

    def test_load_missing_baseline(self) -> None:
        baseline = load_baseline("/nonexistent/path.json")
        assert baseline == []

    def test_custom_eval_set(self) -> None:
        """Load a custom eval set from a temp file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps({"question": "Q1", "ground_truth": "A1", "kb_id": "kb_1"}) + "\n")
            tmp_file = f.name
        try:
            cases = load_eval_cases(tmp_file)
            assert len(cases) == 1
            assert cases[0].question == "Q1"
        finally:
            import os
            os.unlink(tmp_file)


class TestBaselineComparison:
    def test_no_regression(self) -> None:
        """All scores match baseline → no regressions."""
        results = [EvalResult(
            question="Q1",
            metrics={
                "faithfulness": 0.90,
                "answer_correctness": 0.88,
                "citation_accuracy": 0.91,
            },
        )]
        baseline = [{"question": "Q1", "faithfulness": 0.90, "answer_correctness": 0.88, "citation_accuracy": 0.91}]
        regressions = compare_baseline(results, baseline)
        assert regressions == []

    def test_single_metric_drop_flagged(self) -> None:
        """Drop >0.05 on one metric → flagged."""
        results = [EvalResult(
            question="Q1",
            metrics={"faithfulness": 0.80, "answer_correctness": 0.88},
        )]
        baseline = [{"question": "Q1", "faithfulness": 0.90, "answer_correctness": 0.88}]
        regressions = compare_baseline(results, baseline, tolerance=0.05)
        assert len(regressions) == 1
        assert regressions[0]["metric"] == "faithfulness"
        assert regressions[0]["current"] == 0.80
        assert regressions[0]["baseline"] == 0.90
        assert regressions[0]["delta"] == pytest.approx(0.10)

    def test_drop_within_tolerance_not_flagged(self) -> None:
        """Drop ≤ tolerance → not flagged."""
        results = [EvalResult(
            question="Q1",
            metrics={"faithfulness": 0.87, "answer_correctness": 0.88},
        )]
        baseline = [{"question": "Q1", "faithfulness": 0.90, "answer_correctness": 0.88}]
        regressions = compare_baseline(results, baseline, tolerance=0.05)
        assert regressions == []

    def test_hallucination_inverted(self) -> None:
        """Hallucination: lower is better. Increase > tolerance → flagged."""
        results = [EvalResult(
            question="Q1",
            metrics={"hallucination": 0.30},
        )]
        baseline = [{"question": "Q1", "hallucination": 0.10}]
        regressions = compare_baseline(results, baseline, tolerance=0.05)
        assert len(regressions) == 1
        assert regressions[0]["metric"] == "hallucination"
        assert regressions[0]["current"] == 0.30
        assert regressions[0]["baseline"] == 0.10

    def test_hallucination_decrease_not_flagged(self) -> None:
        """Hallucination decrease is an improvement → not flagged."""
        results = [EvalResult(
            question="Q1",
            metrics={"hallucination": 0.05},
        )]
        baseline = [{"question": "Q1", "hallucination": 0.10}]
        regressions = compare_baseline(results, baseline, tolerance=0.05)
        assert regressions == []

    def test_nan_metrics_skipped(self) -> None:
        """NaN metrics are skipped (not gated)."""
        results = [EvalResult(
            question="Q1",
            metrics={"faithfulness": math.nan, "answer_correctness": 0.88},
        )]
        baseline = [{"question": "Q1", "faithfulness": 0.90, "answer_correctness": 0.88}]
        regressions = compare_baseline(results, baseline)
        assert regressions == []

    def test_missing_question_in_baseline_skipped(self) -> None:
        """Question not in baseline → skipped."""
        results = [EvalResult(question="Q_new", metrics={"faithfulness": 0.50})]
        baseline = [{"question": "Q1", "faithfulness": 0.90}]
        regressions = compare_baseline(results, baseline)
        assert regressions == []


class TestLocalEvaluator:
    async def test_citation_accuracy_scored(self) -> None:
        """LocalEvaluator computes citation_accuracy and answer_correctness."""
        evaluator = LocalEvaluator()
        case = EvalCase(
            question="What is RAG?",
            ground_truth="Retrieval Augmented Generation",
            kb_id="kb_1",
            expected_citations=["chk_1"],
        )
        result = _make_result(
            answer="RAG is Retrieval Augmented Generation",
            citation_ids=["chk_1"],
        )
        eval_result = await evaluator.evaluate(case, result)
        assert eval_result.metrics["citation_accuracy"] == pytest.approx(1.0)
        assert eval_result.metrics["answer_correctness"] == 1.0

    async def test_no_expected_citations_nan(self) -> None:
        """No expected citations → citation_accuracy is NaN."""
        evaluator = LocalEvaluator()
        case = EvalCase(question="Q", ground_truth="A", kb_id="kb_1")
        result = _make_result(answer="answer", citation_ids=["chk_1"])
        eval_result = await evaluator.evaluate(case, result)
        assert math.isnan(eval_result.metrics["citation_accuracy"])

    async def test_ground_truth_not_in_answer(self) -> None:
        """Ground truth not in answer → answer_correctness=0."""
        evaluator = LocalEvaluator()
        case = EvalCase(question="Q", ground_truth="specific answer", kb_id="kb_1")
        result = _make_result(answer="completely different", citation_ids=[])
        eval_result = await evaluator.evaluate(case, result)
        assert eval_result.metrics["answer_correctness"] == 0.0


class TestRunEval:
    async def test_run_eval_produces_report(self) -> None:
        """Full eval run produces a report with results."""
        evaluator = LocalEvaluator()

        async def query_fn(question: str, kb_id: str) -> QueryResult:
            # Return an answer containing the ground truth text
            return _make_result(
                answer=f"{question} — the answer is: {question} includes the key info",
                citation_ids=["chk_001", "chk_002"] if "LightRAG" in question else [],
            )

        report = await run_eval(evaluator, query_fn)
        assert report.cases == 3
        assert len(report.results) == 3
        assert report.results[0].question.startswith("LightRAG")

    async def test_run_eval_with_baseline_regression(self) -> None:
        """Eval with a deliberate regression in citation_accuracy."""
        evaluator = LocalEvaluator()

        async def query_fn(question: str, kb_id: str) -> QueryResult:
            # Include ground truth in answer so answer_correctness is OK,
            # but return wrong citations → low citation_accuracy
            return _make_result(
                answer=f"{question} — the correct answer to this question",
                citation_ids=["chk_wrong_1", "chk_wrong_2"],
            )

        report = await run_eval(evaluator, query_fn)
        assert report.passed is False
        assert any(
            r["metric"] == "citation_accuracy" for r in report.regressions
        )

    async def test_run_eval_summary(self) -> None:
        """Report summary aggregates metrics."""
        evaluator = LocalEvaluator()

        async def query_fn(question: str, kb_id: str) -> QueryResult:
            return _make_result(
                answer=f"{question} — detailed answer text",
                citation_ids=[],
            )

        report = await run_eval(evaluator, query_fn)
        summary = report.summary()
        assert summary["cases"] == 3
        assert "citation_accuracy" in summary["metrics"]
        assert "passed" in summary
