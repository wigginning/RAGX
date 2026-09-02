"""Citation accuracy metric (10-observability.md §10.4.2).

Self-implemented: checks whether the cited chunks actually support the
answer. Uses the ``expected_citations`` field from the eval case to
compute precision/recall against the actual citations in the result.

No external dependencies — works in the lite profile.
"""

from __future__ import annotations

import math

from ragx.core.models import QueryResult


def compute_citation_accuracy(
    result: QueryResult,
    expected_citations: list[str],
) -> float:
    """Compute citation accuracy score in [0, 1].

    If ``expected_citations`` is empty (not provided in the eval case),
    returns ``math.nan`` (metric not applicable — excluded from gating).

    Score = harmonic mean of precision and recall over chunk IDs:
    * Precision = |actual ∩ expected| / |actual|
    * Recall    = |actual ∩ expected| / |expected|
    * F1 = 2 * P * R / (P + R) (harmonic mean)

    This is a self-contained metric — both ragas and deepeval backends
    include it identically (§10.4.2).
    """
    if not expected_citations:
        return math.nan

    actual_ids = {c.chunk_id for c in result.citations}
    expected_set = set(expected_citations)

    if not actual_ids:
        # No citations produced — recall is 0, but precision is undefined.
        # Score = 0 (no correct citations).
        return 0.0

    intersection = actual_ids & expected_set
    tp = len(intersection)
    precision = tp / len(actual_ids)
    recall = tp / len(expected_set)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1
