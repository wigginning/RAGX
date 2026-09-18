# RAGX

> A **plugin-based, production-grade, fully observable** open-source RAG engine.
> Multimodal "description-as-vector" ingestion + Hybrid retrieval (vector × BM25 × graph) + adaptive Agentic query paths, with a unified SPI plugin layer for parsers / models / stores.

**English** | [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why RAGX?

| Problem | RAGX answer |
|---|---|
| Vendor lock-in (one vector DB, one model) | 7 SPI interfaces — swap any backend without code changes |
| LLM cost explodes | Resilient Router (retry + fallback + circuit breaker) + semantic cache + role-based cheap/heavy model split |
| Multimodal parsing is fragile | Visual-aware chunker + VLM processor pluggable (DeepDoc / MinerU) |
| No observability | OpenTelemetry spans + Prometheus metrics + cost ledger + RAG Trace replay |
| Slow iteration on prompts | Versioned prompts (`*.v1.yaml`) + per-kb overrides + eval regression gates (`make eval`) |

## Five-minute quickstart (lite profile, zero dependencies)

```bash
pip install -e ".[lite]"
python -c "from ragx.api.app import create_app; print(create_app().title)"
# -> RAGX

# or run the server
ragx-serve
# -> open http://127.0.0.1:8000/v1/health
```

The lite profile uses:
- **Parser**: text/markdown
- **Embedder**: deterministic hash (zero dep)
- **Vector store**: SQLite (`sqlite-vec`)
- **Graph store**: NetworkX (in-memory)
- **LLM**: plug your OpenAI-compatible endpoint

## Full profile (ES + Neo4j + MinIO + PG + Redis + Grafana)

```bash
pip install -e ".[full]"
docker compose -f deploy/compose/full.yml up -d
python scripts/smoke_full.py
```

## Architecture

```text
Client ──▶ API Gateway (Auth · RateLimit · Tenant)
              │
              ├─▶ Query API   ──▶ Fast / Standard / Agentic
              │                   │
              │                   ├─▶ Semantic Cache
              │                   ├─▶ Hybrid Retriever (dense + BM25 + graph)
              │                   └─▶ Resilient LLM Router (Retry · Fallback · CB)
              │
              └─▶ Ingest API  ──▶ Task Queue (Redis Streams)
                                      │
                                      ├─▶ Parser ──▶ Processor (VLM) ──▶ Chunker
                                      │                                ├─▶ Embedder ─▶ VectorStore
                                      │                                └─▶ KGBuilder  ─▶ GraphStore
                                      └─▶ ObjectStore (Local FS / MinIO)
```

Five layers, downward-only dependencies, seven SPI interfaces:

```text
api/                  L4 access layer
agentic/ retrieval/ ingestion/ chunking/ kg/   L3 domain services
llm/ observability/                              L2 cross-cutting
spi/ plugins/                                     L1 contracts + plugins
core/                                             L0 foundation (no upstream deps)
```

See **[Architecture](docs/architecture.md)** (layering rules, SPI layer, error model) and **[Workflows](docs/workflow.md)** (ingestion & query flows end to end).

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat (SSE streamable) |
| `POST /v1/search` | Pure retrieval (Agent-tool friendly) |
| `POST /v1/documents` | Upload a document (returns `task_id`) |
| `GET  /v1/tasks/{task_id}` | Poll ingestion status |
| `GET  /v1/documents/{id}/chunks` | List chunks (governance) |
| `PUT  /v1/chunks/{id}` | Edit a chunk (optimistic lock) |
| `GET  /v1/health?deep=true` | Deep dependency health check |
| `GET  /v1/metrics` | Prometheus exposition |
| `GET  /v1/mcp/sse` · `POST /v1/mcp/messages` | MCP over SSE (opt-in, see below) |

All responses share a unified error envelope `{error: {code, message, trace_id}}`; error codes are grouped by segment — see the [error model in Architecture](docs/architecture.md#error-model).

## MCP (Model Context Protocol)

Expose RAGX retrieval/generation as standard MCP tools — `ragx_search`,
`ragx_generate`, `ragx_list_kbs` (flow overview: [Workflows §3](docs/workflow.md#3-mcp-tool-flow)).

**stdio** — for local agents / CLI; no server changes required:

```bash
ragx-mcp            # or: python -m ragx.mcp
```

**SSE** — for remote/service integrations; opt-in because it mounts extra routes:

```bash
RAGX_MCP.ENABLED=true RAGX_MCP.TRANSPORT=sse ragx-serve
# GET  /v1/mcp/sse        -> emits an `event: sessionId` on connect
# POST /v1/mcp/messages?sessionId=<sid>
```

| Setting | Env var | Default | Notes |
|---|---|---|---|
| `mcp.enabled` | `RAGX_MCP.ENABLED` | `false` | `false` = no `/v1/mcp/*` routes |
| `mcp.transport` | `RAGX_MCP.TRANSPORT` | `stdio` | preferred transport |
| `mcp.sse_idle_timeout` | `RAGX_MCP.SSE_IDLE_TIMEOUT` | `300` | seconds; `<= 0` disables the idle close |

Each SSE session is scoped to the calling API key's `kb_acl`: out-of-scope
`kb_id` yields a tool-level error and `ragx_list_kbs` only lists authorised
knowledge bases — same tenant isolation as the REST API.

## Development

```bash
# install everything
pip install -e ".[lite,dev]"

# run tests (359 passed / 31 skipped; skips are ES/Qdrant/Neo4j/Milvus contract suites)
pytest -q tests/unit tests/contract tests/integration tests/e2e

# coverage gate (threshold + omit in pyproject [tool.coverage.*])
pytest -q --cov=ragx --cov-fail-under=80 tests/unit tests/contract tests/integration tests/e2e

# lint + type-check
ruff check ragx tests scripts
mypy ragx/core ragx/spi ragx/llm ragx/retrieval ragx/api

# evaluation gate — offline smoke, or ragas with ragx[eval]
make eval-local
make eval            # needs ragx[eval] + judge model + seeded eval corpus

# smoke test the lite stack (upload → query → citations end-to-end)
python scripts/smoke_lite.py
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — layered architecture, SPI plugin layer, concurrency model, error model (**[中文](docs/architecture.zh-CN.md)**)
- [`docs/workflow.md`](docs/workflow.md) — ingestion & query answering flows end to end (**[中文](docs/workflow.zh-CN.md)**)
- [`CHANGELOG.md`](CHANGELOG.md) — release notes

## Versioning

RAGX follows [SemVer](https://semver.org/). The SPI is frozen at v1.0.0; earlier `0.x` may break the SPI between minor versions.

| Version | Status | Highlights |
|---|---|---|
| `0.1.0` | critical path | SPI + lite stack + Standard query + Resilient Router + OTel + `/v1/metrics` |
| `0.2.0` | production | Redis Streams + semantic cache + full compose (ES/Neo4j/MinIO/PG) + auth/audit |
| `0.3.0` | multimodal + graph | DeepDoc/MinerU parsers + VLM cost gates + KG dual-level retrieval + 14 prompts |
| `0.4.0` | agentic + eval | Three-tier query router + LangGraph orchestrator + RAGAS/DeepEval harness |
| `1.0.0` | released 2026-09-03 | MCP (SSE+stdio, wired backend) · Helm · multi-tenant/audit (security-reviewed) · in-process E2E · L4 eval gate (`make eval`; ragas runs nightly) |

## License

MIT — see [LICENSE](LICENSE).
