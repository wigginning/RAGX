# RAGX

> A **plugin-based, production-grade, fully observable** open-source RAG engine.
> Multimodal "description-as-vector" ingestion + Hybrid retrieval (vector × BM25 × graph) + adaptive Agentic query paths, with a unified SPI plugin layer for parsers / models / stores.

[![CI](https://github.com/ragx/ragx/actions/workflows/ci.yml/badge.svg)](https://github.com/ragx/ragx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Why RAGX?

| Problem | RAGX answer |
|---|---|
| Vendor lock-in (one vector DB, one model) | 7 SPI interfaces — swap any backend without code changes |
| LLM cost explodes | Resilient Router (retry + fallback + circuit breaker) + semantic cache + role-based cheap/heavy model split |
| Multimodal parsing is fragile | Visual-aware chunker + VLM processor pluggable (DeepDoc / MinerU) |
| No observability | OpenTelemetry spans + Prometheus metrics + cost ledger + RAG Trace replay |
| Slow iteration on prompts | Versioned prompts (`*.v1.yaml`) + per-kb overrides + eval regression in CI |

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

```
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

Layered architecture (00-overview.md):

```
api/                  L4 access layer
agentic/ retrieval/ ingestion/ chunking/ kg/   L3 domain services
llm/ observability/                              L2 cross-cutting
spi/ plugins/                                     L1 contracts + plugins
core/                                             L0 foundation (no upstream deps)
```

See [`DESIGN.md`](DESIGN.md) and [`docs/design/`](docs/design/) for the full design.

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

All responses share a unified error envelope `{error: {code, message, trace_id}}` and error codes by segment (00-overview.md §0.3):
- `1xxx` access · `2xxx` ingest · `3xxx` chunks · `4xxx` retrieval · `5xxx` graph · `6xxx` LLM · `7xxx` agentic · `9xxx` infra

## MCP (Model Context Protocol)

Expose RAGX retrieval/generation as standard MCP tools — `ragx_search`,
`ragx_generate`, `ragx_list_kbs` (see `docs/design/09-api.md` §9.7).

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

# run tests (342 passed / 31 skipped; skips are ES/Qdrant/Neo4j/Milvus contract suites)
pytest -q tests/unit tests/contract tests/integration

# end-to-end journey (upload → ingest → search → chat with citations, in-process)
pytest -q tests/e2e        # 8 passed

# lint + type-check
ruff check ragx tests
mypy ragx/core ragx/spi ragx/llm ragx/retrieval

# smoke test the lite stack (upload → query → citations end-to-end)
python scripts/smoke_lite.py
```

## Documentation

- [`DESIGN.md`](DESIGN.md) — overall design philosophy
- [`docs/design/00-overview.md`](docs/design/00-overview.md) — layering, errors, global conventions
- [`docs/design/01-spi.md`](docs/design/01-spi.md) — 7 SPI interfaces and contracts
- [`docs/design/06-retrieval.md`](docs/design/06-retrieval.md) — hybrid retrieval and routing
- [`docs/design/08-llm.md`](docs/design/08-llm.md) — Resilient Model Router, cache, ledger
- [`docs/design/09-api.md`](docs/design/09-api.md) — REST contracts, auth, streaming, **MCP** (§9.7)
- [`docs/design/12-prompts.md`](docs/design/12-prompts.md) — all 14 prompt templates v1
- [`docs/TASKS.md`](docs/TASKS.md) — task cards with DoD + verification commands
- [`docs/adr/`](docs/adr/) — Architecture Decision Records
- [`CHANGELOG.md`](CHANGELOG.md) — release notes (BREAKING changes marked)

## Versioning

RAGX follows [SemVer](https://semver.org/). The SPI is frozen at v1.0.0; earlier `0.x` may break the SPI between minor versions.

| Version | Status | Highlights |
|---|---|---|
| `0.1.0` | critical path | SPI + lite stack + Standard query + Resilient Router + OTel + `/v1/metrics` |
| `0.2.0` | production | Redis Streams + semantic cache + full compose (ES/Neo4j/MinIO/PG) + auth/audit |
| `0.3.0` | multimodal + graph | DeepDoc/MinerU parsers + VLM cost gates + KG dual-level retrieval + 14 prompts |
| `0.4.0` | agentic + eval | Three-tier query router + LangGraph orchestrator + RAGAS/DeepEval harness |
| `1.0.0` | enterprise | MCP + Helm + multi-tenant + audit + complete E2E |

## Design deviations (recorded per `docs/TASKS.md` §6)

1. **`FilterExpr` operator list vs `">="` examples** — `02-core.md §2.1.3` lists textual ops (`eq/ge/...`), but every example in 02/06/09 uses `">="`. Both forms are accepted and normalised to the canonical ops.
2. **BM25-unsupported handling** — `01-spi.md §1.3` says a missing capability fails startup with `9003`; `11-plugins-builtin.md §11.0` says the retrieval layer auto-skips the route. Implemented as a **soft runtime skip**; hard failure only for `kg_enabled` without a `graph_store` and for `supports_filter=false`.
3. **SQLite BM25 tokenizer** — the default FTS5 `unicode61` tokenizer does not segment CJK, so the FTS table uses the `trigram` tokenizer (substring recall for Chinese; ≥3-char minimum query length).
4. **`LLMRole` location** — lives in `core/roles.py` (both `spi.ChatRequest` and `core.settings` need it); `llm/roles.py` re-exports it.
5. **Lite embedder default** — `11-plugins-builtin.md §11.3.2` lists `st` (sentence-transformers) as the lite default; the critical path requires a zero-dependency install, so `hash` is the lite default and `st` is available behind `ragx[embed-st]`.
6. **Coverage gate scope** — `docs/TASKS.md §4` L1 targets `pytest --cov=ragx --cov-fail-under=80`. `ragx/plugins` (optional third-party adapters for ES/Qdrant/Neo4j/MinIO and heavy parsers/embedders) cannot execute in the dependency-light CI, dragging whole-package coverage to ~73%. The gate therefore omits `ragx/plugins/*` (measured ~84% without it; thresholds in `pyproject.toml [tool.coverage.*]`); plugin behaviour is still exercised by the contract suites wherever a service is present.
7. **`make eval-update`** — `10-observability.md §10.4.3` and `docs/TASKS.md §4` reference a Makefile target; the Makefile now exists (`make eval` / `make eval-local` / `make eval-update`, backed by `scripts/eval_l4.py`), and the L4 gate runs offline with `--backend local` or against a judge model with `ragx[eval]`.

## License

MIT — see [LICENSE](LICENSE).
