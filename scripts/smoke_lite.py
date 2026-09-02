#!/usr/bin/env python3
"""RAGX lite-profile smoke script (RX-INF-01 DoD).

Flow: upload a markdown doc -> wait for the ingest task to reach DONE
(auto-consume processes it in-process) -> search -> chat and assert the answer
carries citations referencing the uploaded document.

Usage:
    docker compose -f deploy/compose/lite.yml up -d --build
    python scripts/smoke_lite.py [--base-url http://localhost:18000]

Exit code 0 on success; non-zero (with a readable message) on failure.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://localhost:18000"
SAMPLE = """# RAGX 冒烟测试

RAGX 是一个分层插件化的检索增强生成平台，使用向量检索与知识图谱提升问答质量。

## 背景

它通过 dense + BM25 混合检索、RRF 融合与引用标注，输出带 [n] 引用的答案。
"""


class SmokeFailure(RuntimeError):
    pass


def _request(
    base: str, method: str, path: str, *, data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, object]:
    url = base + path
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return resp.status, _maybe_json(body)
    except urllib.error.HTTPError as e:
        raise SmokeFailure(f"HTTP {e.code} on {method} {path}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise SmokeFailure(f"unreachable {url}: {e.reason}")


def _maybe_json(body: bytes) -> object:
    try:
        import json

        return json.loads(body.decode("utf-8"))
    except Exception:
        return body.decode("utf-8", errors="replace")


def _wait_done(base: str, task_id: str, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status, body = _request(base, "GET", f"/v1/tasks/{task_id}")
        assert isinstance(body, dict)
        st = body.get("status")
        if st == "done":
            return
        if st == "failed":
            raise SmokeFailure(f"task {task_id} failed: {body.get('error')}")
        time.sleep(1.0)
    raise SmokeFailure(f"task {task_id} did not reach done within {timeout_s}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument(
        "--kb-id", default="default",
        help="kb to upload to; 'default' matches the app-level query_service wiring",
    )
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # 1) health
    status, body = _request(base, "GET", "/v1/health")
    print(f"[1/4] health -> {status} {body}")
    assert status == 200

    # 2) upload
    boundary = "----ragxsmoke"
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="smoke.md"\r\n'
        f"Content-Type: text/markdown\r\n\r\n"
        f"{SAMPLE}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="kb_id"\r\n\r\n'
        f"{args.kb_id}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="metadata"\r\n\r\n'
        f'{{}}\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    status, body = _request(
        base, "POST", "/v1/documents",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    print(f"[2/4] upload -> {status} {body}")
    assert status == 202, "upload must return 202 Accepted"
    assert isinstance(body, dict)
    doc_id = body["doc_id"]
    task_id = body["task_id"]

    # 3) wait for ingest to finish (auto-consume processes it in-process)
    _wait_done(base, task_id)
    print(f"[3/4] ingest task {task_id} -> done")

    # 4a) search must return the ingested chunk
    import json

    search_body = json.dumps({
        "kb_id": args.kb_id, "query": "向量检索", "top_k": 5,
    }).encode()
    status, body = _request(
        base, "POST", "/v1/search",
        data=search_body,
        headers={"Content-Type": "application/json"},
    )
    print(f"[4/4] search -> {status}")
    assert status == 200
    assert isinstance(body, dict)
    results = body.get("results") or []
    assert results, "search must return at least one result"
    hit_doc_ids = {r.get("doc_id") for r in results}
    assert doc_id in hit_doc_ids, f"search must return a chunk of {doc_id}, got {hit_doc_ids}"

    # 4b) chat must carry citations referencing the uploaded document
    chat_body = json.dumps({
        "model": "ragx-default",
        "stream": False,
        "messages": [{"role": "user", "content": "什么是 RAGX?"}],
        "ragx": {"kb_id": args.kb_id, "mode": "auto"},
    }).encode()
    status, body = _request(
        base, "POST", "/v1/chat/completions",
        data=chat_body,
        headers={"Content-Type": "application/json"},
    )
    print(f"[4/5] chat -> {status}")
    assert status == 200
    assert isinstance(body, dict)
    citations = (body.get("ragx") or {}).get("citations") or []
    assert citations, "chat answer must carry citations"
    cit_doc_ids = {c.get("doc_id") for c in citations}
    assert doc_id in cit_doc_ids, f"citations must reference {doc_id}, got {cit_doc_ids}"

    print("SMOKE OK: upload -> ingest -> search -> chat with citations")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SmokeFailure as e:
        print(f"SMOKE FAILED: {e}", file=sys.stderr)
        sys.exit(1)
