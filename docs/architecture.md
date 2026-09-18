# Architecture

> How RAGX is structured: a five-layer architecture with a strict downward-only
> dependency rule, a unified SPI plugin layer, and cross-cutting LLM /
> observability services.

**English** | [简体中文](architecture.zh-CN.md)

## Layered architecture

```text
┌─────────────────────────────────────────────┐
│ api/          FastAPI routes, middleware, MCP │  Layer 4: Access
├─────────────────────────────────────────────┤
│ agentic/ retrieval/ ingestion/ chunking/ kg/ │  Layer 3: Domain services
├─────────────────────────────────────────────┤
│ llm/ observability/                          │  Layer 2: Cross-cutting
├─────────────────────────────────────────────┤
│ spi/  plugins/                               │  Layer 1: Contracts + plugins
├─────────────────────────────────────────────┤
│ core/         domain models, errors, config  │  Layer 0: Foundation
└─────────────────────────────────────────────┘
```

**Dependency rules** (enforced via import-linter):

1. Dependencies point downward only: `api → domain services → cross-cutting → spi → core`.
   No reverse imports, no direct cross-domain calls between domain services.
2. Domain services cooperate in exactly two ways:
   - passing data through domain models in `core/` (behaviour-free);
   - calling `spi/` interfaces (e.g. retrieval calls `GraphStore`, never imports
     the `kg/` module internals).
3. `plugins/` implement `spi/` interfaces and may depend only on `spi + core +
   third-party SDKs` — never on the domain service layer.
4. `observability/` and `llm/` are cross-cutting: layers 3 and 4 may depend on
   them; they themselves depend only on `spi + core`.
5. Third-party frameworks are quarantined: LangGraph only inside `agentic/`;
   FastAPI only inside `api/`; Redis/Celery clients only inside `ingestion/`
   and `plugins/`.

## End-to-end request topology

```text
Client ──▶ API Gateway (Auth · RateLimit · Tenant)
              │
              ├─▶ Query API   ──▶ Fast / Standard / Agentic
              │                   │
              │                   ├─▶ Semantic Cache
              │                   ├─▶ Hybrid Retriever (dense + BM25 + graph)
              │                   └─▶ Resilient LLM Router (Retry · Fallback · Circuit breaker)
              │
              └─▶ Ingest API  ──▶ Task Queue (Redis Streams)
                                      │
                                      ├─▶ Parser ──▶ Processor (VLM) ──▶ Chunker
                                      │                                ├─▶ Embedder ─▶ VectorStore
                                      │                                └─▶ KGBuilder  ─▶ GraphStore
                                      └─▶ ObjectStore (Local FS / MinIO)
```

## SPI plugin layer

Seven interfaces form the extension surface. Any backend can be swapped by
implementing the corresponding interface and registering the plugin — no
application-code changes required:

| Interface | Responsibility | Built-in options |
|---|---|---|
| Parser | File → structured content | text/markdown (lite), DeepDoc / MinerU (heavy) |
| Processor | Enrich content with VLM | optional, behind `vlm_enabled` |
| Chunker | Content → chunks | visual-aware chunker |
| Embedder | Chunks → vectors | deterministic hash (lite, zero-dep); sentence-transformers (`ragx[embed-st]`) |
| VectorStore | Vector storage & ANN search | SQLite `sqlite-vec` (lite); Elasticsearch / Qdrant / Milvus (full) |
| GraphStore | Knowledge graph storage | NetworkX (in-memory, lite); Neo4j (full) |
| LLM Provider | Chat / generation | any OpenAI-compatible endpoint |

## Concurrency model

| Boundary | Convention |
|---|---|
| Ingestion pipeline | Fully async, task-queue driven; the API exposes only "submit task" + "poll status" |
| Query pipeline | Async handlers with synchronous request semantics; SSE streaming; parallel multi-recall via `asyncio.gather` |
| SPI interfaces | All methods are `async def`; synchronous SDKs are wrapped with `asyncio.to_thread` inside plugins |
| LLM calls | Always go through `llm.router`; business code never instantiates provider clients directly |

## Error model

All errors share one envelope: `{error: {code, message, trace_id}}`. Codes are
grouped by segment; HTTP status codes only express the coarse category
(400/401/403/404/409/429/500/503).

| Segment | Area | Examples |
|---|---|---|
| 1xxx | access | 1001 validation, 1002 unauthenticated, 1003 forbidden, 1004 rate-limited |
| 2xxx | ingestion | 2001 parse failure, 2002 unsupported format, 2003 task not found, 2004 duplicate document |
| 3xxx | chunks | 3001 chunk not found, 3002 edit conflict |
| 4xxx | retrieval | 4001 KB not found, 4002 empty recall, 4003 invalid filter expression |
| 5xxx | graph | 5001 graph build failure, 5002 graph store unavailable, 5003 extraction schema mismatch |
| 6xxx | LLM | 6001 all providers failed, 6002 circuit open, 6003 token budget exceeded, 6004 cache backend down |
| 7xxx | agentic | 7001 planning failure, 7002 task execution exhausted, 7003 verification failed (falls back to Standard) |
| 9xxx | infrastructure | 9001 storage unavailable, 9002 queue unavailable, 9003 configuration error |

## Global conventions

- **IDs**: typed prefixes (`doc_` / `chk_` / `ent_` / `rel_` / `task_` / `kb_`) + ULID (sortable, URL-safe).
- **Time**: UTC everywhere, serialized as ISO-8601.
- **trace_id**: generated at the API entry, propagated across ingestion and query paths; present in logs, spans, the cost ledger, and RAG traces.
- **Configuration**: three-level override — global settings ← per-knowledge-base config ← per-request override.
- **Feature flags**: `kg_enabled` / `vlm_enabled` / `agentic_enabled` / `cache_enabled` / `av_enabled`, scoped per knowledge base.
- **Stack**: Python ≥ 3.11, Pydantic v2, full type annotations, `mypy --strict` on core modules.
