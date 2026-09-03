#!/usr/bin/env python
"""End-to-end smoke test for the RAGX full compose stack (RX-INF-01).

Uploads a short document, polls until ingestion reaches DONE, then issues a
retrieval + chat query and asserts the response carries citations.

Usage:

    python scripts/smoke_full.py [--api http://localhost:18000]

The script assumes the services in deploy/compose/full.yml are running and
reachable. It is intentionally chatty: every step prints its status so the
output can be grepped in CI logs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx


def _post(client: httpx.Client, path: str, *, files: Any = None, data: Any = None,
          json: Any = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    r = client.post(path, files=files, data=data, json=json, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json() if r.content else {}


def _get(client: httpx.Client, path: str) -> dict[str, Any]:
    r = client.get(path, timeout=10)
    r.raise_for_status()
    return r.json() if r.content else {}


def _wait_task_done(client: httpx.Client, task_id: str, deadline_s: float = 60) -> dict[str, Any]:
    end = time.time() + deadline_s
    last: dict[str, Any] = {}
    while time.time() < end:
        last = _get(client, f"/v1/tasks/{task_id}")
        if last.get("status") in {"done", "failed"}:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"task {task_id} did not finish in {deadline_s}s; last={last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:18000")
    args = ap.parse_args()

    with httpx.Client(base_url=args.api) as client:
        print(f"[smoke] target: {args.api}")
        h = _get(client, "/v1/health?deep=true")
        print(f"[smoke] health: {h}")
        # The full stack has ES/Neo4j/MinIO/Redis/PG; the deep health probe
        # checks vector store + metadata DB. Pass on the critical-path
        # checks at minimum.
        assert h.get("status") in {"ok", "degraded"}, h

        # 1. Upload a document.
        sample_text = (
            b"RAGX is a plugin-based RAG engine. "
            b"It uses a 7-interface SPI for parsers, embedders, vector stores, "
            b"graph stores, LLM providers, rerankers, and processors. "
            b"It supports OpenAI-compatible APIs and exposes an MCP server."
        )
        upload = _post(
            client,
            "/v1/documents",
            files={"file": ("hello.txt", sample_text, "text/plain")},
            data={"kb_id": "default", "metadata": json.dumps({"source": "smoke"})},
        )
        print(f"[smoke] upload: {upload}")
        task_id = upload["task_id"]

        # 2. Poll until DONE.
        final = _wait_task_done(client, task_id)
        print(f"[smoke] task final: {final}")
        if final.get("status") != "done":
            print(f"[smoke] task did not succeed: {final}", file=sys.stderr)
            return 2

        # 3. Retrieve.
        search = _post(
            client,
            "/v1/search",
            json={"kb_id": "default", "query": "What is RAGX?", "top_k": 3},
        )
        print(f"[smoke] search: {search}")
        assert search["results"], "search returned empty results"

        # 4. Chat (SSE disabled — non-streaming for simplicity in CI).
        chat = _post(
            client,
            "/v1/chat/completions",
            json={
                "model": "ragx-default",
                "messages": [{"role": "user", "content": "What is RAGX?"}],
                "stream": False,
                "ragx": {"kb_id": "default", "mode": "standard"},
            },
        )
        print(f"[smoke] chat: {json.dumps(chat, ensure_ascii=False)[:400]}")
        assert chat["choices"], "chat returned no choices"
        assert chat["ragx"]["citations"], "chat response has no citations"
        assert chat["ragx"]["mode"] in {"fast", "standard", "agentic"}

        # 5. Metrics endpoint smoke.
        r = client.get("/v1/metrics")
        assert r.status_code == 200, f"/v1/metrics returned {r.status_code}"
        assert "ragx_query_total" in r.text
        print("[smoke] /v1/metrics: ok")

        print("[smoke] OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
