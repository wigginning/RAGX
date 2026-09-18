# RAGX developer targets.
#
# Primary tooling is pytest / ruff (see README "Development"). These targets
# wrap the L4 evaluation gate (10-observability.md §10.4 / docs/TASKS.md §4).

PY ?= python

.PHONY: eval eval-local eval-update

## L4 evaluation gate (ragas backend by default).
## Needs `pip install -e ".[eval]"`, a configured judge model, and the eval
## KB(s) from tests/eval/ragx_eval.jsonl ingested. BACKEND=deepeval switches.
eval:
	$(PY) scripts/eval_l4.py --backend $(or $(BACKEND),ragas)

## Offline L4 smoke: no external libraries / judge model / corpus needed.
## Scores citation_accuracy + an answer_correctness heuristic.
eval-local:
	$(PY) scripts/eval_l4.py --backend local

## Refresh tests/eval/baseline.json from the latest run.
## Manual only — after verifying an improvement (docs/TASKS.md §4:
## "禁止自动覆盖"; commit message `eval: bump baseline`).
eval-update:
	$(PY) scripts/eval_l4.py --backend $(or $(BACKEND),ragas) --update-baseline
