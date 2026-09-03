"""Evaluation framework (10-observability.md §10.4).

Dual-backend evaluation (ragas | deepeval) with a self-implemented
``citation_accuracy`` metric. Baseline comparison gates CI: single-metric
drop >0.05 fails the gate.

Backends are lazy-imported — ragas/deepeval are optional dependencies.
The citation_accuracy metric is self-contained (no heavy deps).
"""

from ragx.observability.eval.base import EvalCase, EvalResult, Evaluator
from ragx.observability.eval.citation_accuracy import compute_citation_accuracy
from ragx.observability.eval.run_eval import (
    EvalReport,
    build_evaluator,
    compare_baseline,
    run_eval,
)

__all__ = [
    "EvalCase",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "build_evaluator",
    "compare_baseline",
    "compute_citation_accuracy",
    "run_eval",
]
