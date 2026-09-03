#!/usr/bin/env python3
"""L4 evaluation runner (10-observability.md §10.4).

Runs the golden eval set (``tests/eval/ragx_eval.jsonl``) through a live
in-process RAGX app and compares the scores against
``tests/eval/baseline.json``. A single-metric drop beyond ``--tolerance``
fails the gate (exit 1); hallucination is inverted (a rise fails).

Usage::

    python scripts/eval_l4.py [--backend ragas|deepeval|local] \\
        [--judge-model MODEL] [--kb KB] [--update-baseline] [--tolerance 0.05]

* ``--backend local`` (default) needs no external libraries and no LLM judge —
  it scores ``citation_accuracy`` (self-implemented) plus an
  ``answer_correctness`` heuristic, so the L4 loop can be exercised offline.
* ``ragas`` / ``deepeval`` lazy-import their libraries (install
  ``pip install -e ".[eval]"``); without them those metrics report NaN and the
  runner exits 2 (infra error) rather than silently passing.
* The eval KB(s) referenced by the golden set must already be ingested and
  indexed (kb_id / expected chunk ids come from the canonical eval corpus).
* ``--update-baseline`` writes the per-question results over baseline.json
  (design: manual ``make eval-update`` only, §10.4.3 — never automatic).

Exit codes: 0 = pass (no regressions) · 1 = regressions · 2 = infra error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from ragx.api.app import create_app
from ragx.core.ids import new_id
from ragx.core.settings import Settings
from ragx.observability.eval.run_eval import (
    DEFAULT_BASELINE,
    DEFAULT_EVAL_SET,
    DEFAULT_TOLERANCE,
    build_evaluator,
    run_eval,
)

logging.basicConfig(level=logging.WARNING)

_METRIC_LABELS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_recall": "context_recall",
    "context_precision": "context_precision",
    "answer_correctness": "answer_correctness",
    "hallucination": "hallucination(low=good)",
    "citation_accuracy": "citation_accuracy",
}


def _query_fn_factory(app):
    """Build an async (question, kb_id) -> QueryResult callable (§10.4.3)."""

    async def query_fn(question: str, kb_id: str):
        embedder = app.state.embedder
        qvec = (await embedder.embed([question]))[0]
        return await app.state.query_service.query(
            question, qvec, kb_id=kb_id, trace_id=new_id("trace_")
        )

    return query_fn


def _fmt(v: float) -> str:
    return "  n/a" if v != v else f"{v:5.3f}"


def _print_report(report) -> None:
    print("\n=== L4 eval report ===")
    print(f"cases: {report.cases}   backend results: {len(report.results)}")
    if report.results:
        print(f"{'metric':<28} {'mean':>6}")
        for m, v in sorted(report.summary().get("metrics", {}).items()):
            print(f"{_METRIC_LABELS.get(m, m):<28} {_fmt(v)}")
    if report.regressions:
        print("\nREGRESSIONS:")
        for r in report.regressions:
            print(
                f"  {r['question'][:40]:<42} {r['metric']:<22} "
                f"cur={r['current']:.3f} base={r['baseline']:.3f} "
                f"delta={r['delta']:.3f}"
            )
    print(f"\nresult: {'PASS' if report.passed else 'FAIL (regressions)'}")


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    app = create_app(settings=settings)
    # Run the app's lifespan (db / stores connect + shutdown) in this loop.
    async with app.router.lifespan_context(app):
        if args.kb:
            kb_ids = [k.strip() for k in args.kb.split(",") if k.strip()]
            for kb_id in kb_ids:
                try:
                    await app.state.db.save_kb(kb_id, {"name": kb_id, "source": "eval"})
                except Exception:  # noqa: BLE001 - kb may already exist
                    pass

        backend = args.backend or settings.evaluation.backend
        judge = args.judge_model or settings.evaluation.judge_model
        evaluator = build_evaluator(backend, judge_model=judge)

        # Non-local backends need their library present and (for ragas) a judge.
        if backend != "local" and getattr(evaluator, "_ragas", None) is None \
                and getattr(evaluator, "_deepeval", None) is None:
            print(
                f"ERROR: backend '{backend}' is not available — "
                f"install `pip install -e \".[eval]\"` and set a judge model.",
                file=sys.stderr,
            )
            return 2

        report = await run_eval(
            evaluator,
            _query_fn_factory(app),
            eval_set_path=args.eval_set,
            baseline_path=args.baseline,
            tolerance=args.tolerance,
        )

    _print_report(report)

    if args.update_baseline:
        baseline_path = Path(args.baseline) if args.baseline else DEFAULT_BASELINE
        rows = []
        for res in report.results:
            row: dict = {"question": res.question}
            row.update(
                {k: round(v, 4) for k, v in res.metrics.items() if v == v}
            )
            rows.append(row)
        baseline_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nbaseline updated: {baseline_path} ({len(rows)} rows)")

    if backend != "local" and not report.results:
        return 2
    return 0 if report.passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="L4 eval runner (§10.4)")
    parser.add_argument("--backend", choices=["local", "ragas", "deepeval"],
                        help="override Settings.evaluation.backend")
    parser.add_argument("--judge-model", help="LLM judge (backend != local)")
    parser.add_argument("--kb", help="comma-separated kb_ids to ensure exist")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    code = asyncio.run(_run(args, settings))
    sys.exit(code)


if __name__ == "__main__":
    main()
