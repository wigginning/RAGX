# Workflows

> The two end-to-end flows of RAGX: document ingestion (async, queue-driven)
> and query answering (hybrid retrieval → generation with citations).

**English** | [简体中文](workflow.zh-CN.md)

## 1. Document ingestion flow

```
Upload ──▶ POST /v1/documents ──▶ Task Queue (Redis Streams) ──▶ Worker pipeline
                                                                        │
   ┌────────────────────────────────────────────────────────────────────┘
   │
   ├─ 1. Parse        file → structured content
   │                  (text/markdown in lite; DeepDoc / MinerU for PDF,
   │                   scanned docs, cross-page tables in full)
   ├─ 2. Process      optional VLM enrichment (image understanding,
   │                  chart/table description — "description-as-vector")
   ├─ 3. Chunk        visual-aware chunking; chunk-level metadata
   ├─ 4. Embed        chunks → vectors (hash embedder in lite)
   ├─ 5. Index        vectors → VectorStore (SQLite-vec / ES / Qdrant / Milvus)
   ├─ 6. Build KG     entity & relation extraction → GraphStore
   │                  (NetworkX / Neo4j; skipped when kg_enabled=false)
   └─ 7. Finalize     object stored (Local FS / MinIO), task status = done
```

Key properties:

- **Fully async**: the API exposes only two synchronous operations —
  `POST /v1/documents` (returns a `task_id`) and
  `GET /v1/tasks/{task_id}` (polls status).
- **Cost gates**: VLM and KG steps are feature-flagged per knowledge base
  (`vlm_enabled` / `kg_enabled`), so expensive steps can be disabled per KB.
- **Incremental re-indexing**: re-uploading a document updates its chunks
  without rebuilding the whole KB; duplicate uploads are rejected (`2004`).
- **Chunk governance**: after ingestion, chunks are inspectable and editable
  (`GET /v1/documents/{id}/chunks`, `PUT /v1/chunks/{id}` with optimistic
  locking) — retrieval always serves the governed chunk state.

## 2. Query answering flow

```
Question ──▶ POST /v1/search (or /v1/chat/completions)
                 │
                 ├─ 0. Gateway          auth → rate limit → tenant scoping
                 ├─ 1. Semantic cache   hit? → return immediately
                 ├─ 2. Query routing    Fast / Standard / Agentic
                 │
                 ├─ 3. Hybrid recall    (parallel)
                 │      ├─ dense:      query → embed → VectorStore ANN
                 │      ├─ sparse:     BM25 / FTS (trigram tokenizer, CJK-friendly)
                 │      └─ graph:      entity linking → subgraph expansion (if kg_enabled)
                 │
                 ├─ 4. Fusion           merge + dedupe + score → top_k chunks
                 ├─ 5. Generation       LLM Router picks a model by role
                 │                      (cheap model for simple steps, heavy for synthesis)
                 │                      retry → fallback → circuit breaker
                 └─ 6. Respond          answer + citations (doc/chunk provenance)
```

### Three query modes

| Mode | Path | Typical use |
|---|---|---|
| **Fast** | cache → single-shot retrieval → short answer | autocomplete, low-latency lookups |
| **Standard** | hybrid recall → fusion → single generation pass with citations | default API mode |
| **Agentic** | LangGraph orchestrator: plan → multi-step retrieval/tool use → verify | complex multi-hop questions; falls back to Standard when verification fails |

### Reliability layers

- **Semantic cache** — repeated/near-duplicate questions skip retrieval and
  generation entirely (`cache_enabled`).
- **Resilient LLM Router** — every LLM call goes through one router that owns
  retry, provider fallback, and circuit breaking; token budgets are enforced
  per request (error `6003`).
- **Graceful degradation** — if BM25 is unavailable the route is skipped at
  runtime; if a graph store is missing, graph recall is skipped; if the Agentic
  pipeline fails verification, it falls back to Standard instead of erroring.
- **Full observability** — a `trace_id` generated at the API entry propagates
  through every span, log line, and cost ledger entry; OpenTelemetry spans and
  Prometheus metrics (`/v1/metrics`) cover both flows.

## 3. MCP tool flow

RAGX capabilities are also exposed as standard MCP tools — `ragx_search`,
`ragx_generate`, `ragx_list_kbs` — over stdio (local agents/CLI) or SSE
(remote/service integrations). Each SSE session is scoped to the calling API
key's knowledge-base ACL; out-of-scope KB ids yield tool-level errors, the
same tenant isolation as the REST API. See the
[README MCP section](../README.md#mcp-model-context-protocol) for endpoints
and configuration.
