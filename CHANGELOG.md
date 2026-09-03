# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [SemVer](https://semver.org/spec/v2.0.0.html).
Contract-breaking changes are marked **BREAKING** and, per `docs/TASKS.md` §5.1,
are only permitted before `v1.0.0` (which freezes the SPI and REST contracts).

Migration/upgrade notes live in `docs/design/`; the task → release mapping lives
in `docs/TASKS.md` §5.1.

---

## [Unreleased] — targeting `v1.0.0`

Exit criteria (`docs/TASKS.md` §5.1): `F(API-03 多租户审计, INF-03 Helm)` +
MCP 完善 + E2E 全套 + 文档齐备, with L1–L4 green and a passed security review.

### Added

- **MCP server wired to the live backend** (`4705f09`) — `build_mcp_server()`
  connects `ragx_search` / `ragx_generate` / `ragx_list_kbs` to the real
  registry, retriever and metadata store. Previously the three tools shipped as
  inert stubs and returned placeholders.
- **MCP SSE transport** (`4705f09`) — `GET /v1/mcp/sse` +
  `POST /v1/mcp/messages`, opt-in via `mcp.enabled` (off by default, so no extra
  routes are mounted unless requested).
- **MCP stdio launcher** (`4705f09`) — `ragx-mcp` console script /
  `python -m ragx.mcp`, for local agents and CLI use.
- **`MCPConfig`** (`4705f09`) — `enabled` / `transport` / `sse_idle_timeout`.
- **End-to-end test suite** (`aad37c8`) — `tests/e2e/` runs the full journey
  in-process (upload → ingest → search → chat with citations), 8 tests.
- **Helm chart** (`79645af`) — K8s deployment with pluggable middleware
  (ES/Neo4j/MinIO/PG/Redis selectable), plus a `kind` smoke path.
- **RAG Trace capture + replay API** (`787152c`) — `GET /v1/traces` (RX-OBS-04).
- **Semantic cache activation** (`4674349`) — LLM-02 cache now consulted/stored
  at runtime instead of only being constructed.

### Fixed

- **Multi-tenant retrieval returned empty results** (`aad37c8`) — `QueryService`
  was bound to the vector store resolved for `kb_id="default"`, while ingest and
  search resolve plugins per real `kb_id`. Every non-`default` kb therefore
  retrieved from a different (empty) store and chat produced no citations.
  Retrievers are now resolved per kb and cached.
- **Cross-tenant reads on id-addressed routes** (`de0d8e4`) — `GET /v1/tasks/{id}`
  and the chunk list/PUT/PATCH endpoints now enforce the caller's `kb_acl`
  (403 with error code `1003`) instead of trusting the id alone.
- **Idle SSE connections could hang forever** (`4705f09`) — the SSE stream
  blocked on an empty queue and was never reaped. It now terminates on client
  disconnect and on `mcp.sse_idle_timeout` (default 300s, `<= 0` disables).

### Security

- MCP tool calls honour the calling key's `kb_acl`: an out-of-scope `kb_id`
  returns a tool-level error rather than data, and `ragx_list_kbs` only lists
  authorised knowledge bases — matching REST isolation.

---

## Earlier releases

Per-release notes before this file was introduced are summarised in the version
table in [`README.md`](README.md#versioning). The SPI and REST contracts are
frozen from `v1.0.0` onwards.
