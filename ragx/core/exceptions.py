"""Exception tree (02-core.md §2.3, error bands 00-overview.md §0.3).

Every domain exception carries ``code: int``, ``message: str``,
``details: dict``, ``trace_id: str | None``. The API layer maps them to the
unified ``{error: {code, message, trace_id}}`` body (09-api.md §9.1.2).

Rule (02-core.md §2.3): third-party exceptions raised inside a plugin must be
translated at the plugin boundary (keeping ``__cause__``); raw exceptions must
never cross the SPI.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RAGXError",
    "ConfigError",
    "AuthError",
    "RateLimitError",
    "IngestError",
    "ParseError",
    "UnsupportedFormatError",
    "TaskNotFoundError",
    "DuplicateDocumentError",
    "CheckpointCorruptError",
    "ChunkError",
    "ChunkNotFoundError",
    "EditConflictError",
    "ChunkTooLargeError",
    "RetrievalError",
    "KBNotFoundError",
    "FilterValidationError",
    "GraphError",
    "GraphBuildError",
    "GraphStoreUnavailableError",
    "ExtractionSchemaError",
    "LLMError",
    "AllProvidersFailedError",
    "CircuitOpenError",
    "BudgetExceededError",
    "CacheBackendError",
    "StructuredParseError",
    "AgenticError",
    "InfraError",
    "PluginContractError",
    "PluginTimeoutError",
    "StoreUnavailableError",
    "QueueUnavailableError",
]


class RAGXError(Exception):
    """Base class of every RAGX domain error.

    Parameters
    ----------
    code:
        Numeric error code from the 00-overview.md band table.
    message:
        Human readable (Chinese or English) description.
    details:
        Structured context. Never leaked to API clients (09-api.md §9.1.2);
        kept for logs / audit only.
    trace_id:
        Propagated trace id when known at raise time.
    """

    #: default code; subclasses override
    default_code: int = 9000

    def __init__(
        self,
        message: str,
        *,
        code: int | None = None,
        details: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.code: int = code if code is not None else self.default_code
        self.message: str = message
        self.details: dict[str, Any] = dict(details or {})
        self.trace_id: str | None = trace_id
        super().__init__(message)

    def to_dict(self, *, include_details: bool = False) -> dict[str, Any]:
        """Serialise for the API error envelope (details stripped by default)."""
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.trace_id:
            body["trace_id"] = self.trace_id
        if include_details:
            body["details"] = self.details
        return body


class ConfigError(RAGXError):
    """9003 - configuration error (missing plugin config, bad override key,
    prompt placeholder mismatch, profile validation failure)."""

    default_code = 9003


class AuthError(RAGXError):
    """1002 unauthenticated / 1003 not authorised (09-api.md §9.2)."""

    default_code = 1002


class RateLimitError(RAGXError):
    """1004 rate limit / daily budget cap reached."""

    default_code = 1004


class ValidationError(RAGXError):
    """1001 request validation failure (Pydantic errors are mapped here)."""

    default_code = 1001


# ---------------------------------------------------------------------------
# 2xxx ingestion
# ---------------------------------------------------------------------------
class IngestError(RAGXError):
    """2xxx ingestion band."""

    default_code = 2000


class ParseError(IngestError):
    """2001 parser failure - retryable (03-ingestion.md §3.2.3)."""

    default_code = 2001


class UnsupportedFormatError(IngestError):
    """2002 mimetype not supported - not retryable."""

    default_code = 2002


class TaskNotFoundError(IngestError):
    """2003 task id not found - HTTP 404."""

    default_code = 2003


class DuplicateDocumentError(IngestError):
    """2004 document already ingested - soft notice, HTTP 200."""

    default_code = 2004


class CheckpointCorruptError(IngestError):
    """2005 checkpoint JSON unreadable - manual replay (03-ingestion.md §3.9)."""

    default_code = 2005


# ---------------------------------------------------------------------------
# 3xxx chunking
# ---------------------------------------------------------------------------
class ChunkError(RAGXError):
    """3xxx chunk governance band."""

    default_code = 3000


class ChunkNotFoundError(ChunkError):
    """3001 chunk not found - HTTP 404."""

    default_code = 3001


class EditConflictError(ChunkError):
    """3002 optimistic-lock version mismatch - HTTP 409."""

    default_code = 3002


class ChunkTooLargeError(ChunkError):
    """3003 assembled chunk exceeds hard cap and cannot be split."""

    default_code = 3003


# ---------------------------------------------------------------------------
# 4xxx retrieval
# ---------------------------------------------------------------------------
class RetrievalError(RAGXError):
    """4xxx retrieval band."""

    default_code = 4000


class KBNotFoundError(RetrievalError):
    """4001 knowledge base not found - HTTP 404."""

    default_code = 4001


class FilterValidationError(RetrievalError):
    """4003 filter expression is invalid - HTTP 400 (06-retrieval.md §6.6.1)."""

    default_code = 4003


# ---------------------------------------------------------------------------
# 5xxx graph
# ---------------------------------------------------------------------------
class GraphError(RAGXError):
    """5xxx knowledge graph band."""

    default_code = 5000


class GraphBuildError(GraphError):
    """5001 graph build failure - retryable."""

    default_code = 5001


class GraphStoreUnavailableError(GraphError):
    """5002 graph store unreachable."""

    default_code = 5002


class ExtractionSchemaError(GraphError):
    """5003 extracted JSON violates the schema - retry once then skip."""

    default_code = 5003


# ---------------------------------------------------------------------------
# 6xxx LLM
# ---------------------------------------------------------------------------
class LLMError(RAGXError):
    """6xxx LLM band."""

    default_code = 6000


class AllProvidersFailedError(LLMError):
    """6001 every candidate for a role failed - HTTP 502."""

    default_code = 6001


class CircuitOpenError(LLMError):
    """6002 circuit breaker open - fast fail, HTTP 503."""

    default_code = 6002


class BudgetExceededError(LLMError):
    """6003 token budget exceeded (Agentic 100k hard cap)."""

    default_code = 6003


class CacheBackendError(LLMError):
    """6004 semantic cache backend unavailable - degrade to uncached."""

    default_code = 6004


class StructuredParseError(LLMError):
    """Structured output could not be parsed into the requested schema.

    Retryable exactly once by the router (08-llm.md §8.3).
    """

    default_code = 6005


# ---------------------------------------------------------------------------
# 7xxx agentic
# ---------------------------------------------------------------------------
class AgenticError(RAGXError):
    """7xxx agentic band (07-agentic.md)."""

    default_code = 7000


class AgenticPlanError(AgenticError):
    """7001 planner failed - degrade to Standard."""

    default_code = 7001


class AgenticTasksFailedError(AgenticError):
    """7002 every sub-task failed - degrade to Standard."""

    default_code = 7002


class AgenticVerifyFailedError(AgenticError):
    """7003 verification failed - degrade to Standard (HTTP 200)."""

    default_code = 7003


# ---------------------------------------------------------------------------
# 9xxx infrastructure
# ---------------------------------------------------------------------------
class InfraError(RAGXError):
    """9xxx infrastructure band."""

    default_code = 9000


class StoreUnavailableError(InfraError):
    """9001 metadata / vector storage unreachable."""

    default_code = 9001


class QueueUnavailableError(InfraError):
    """9002 task queue unreachable."""

    default_code = 9002


class PluginContractError(InfraError):
    """A plugin violated its SPI contract (plugin bug, not an upstream outage).

    Not retryable (03-ingestion.md §3.2.3): retrying a contract violation
    cannot succeed and would waste provider quota.
    """

    default_code = 9004


class PluginTimeoutError(InfraError):
    """A plugin exceeded its timeout obligation (default 30s, 01-spi.md §1.2)."""

    default_code = 9005
