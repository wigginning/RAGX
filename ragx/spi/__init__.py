"""SPI layer (L1): the seven plugin protocols and the registry (01-spi.md).

The pytest-backed contract suite base classes live in
:mod:`ragx.spi.contracts` and are imported explicitly by tests — they are
*not* re-exported here, because doing so would pull ``pytest`` into the
production import chain (the lite/full API images ship no dev dependencies).
"""

from ragx.spi.interfaces import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DescribeOptions,
    Embedder,
    Entity,
    FilterExpr,
    GraphStore,
    GraphStoreCapabilities,
    LLMCapabilities,
    LLMProvider,
    Parser,
    ParseResult,
    Processor,
    RawDocument,
    Reranker,
    RerankResult,
    ScoredChunk,
    SubGraph,
    TopicSummary,
    VectorStore,
    VectorStoreCapabilities,
)
from ragx.spi.registry import PluginRegistry, default_plugin

__all__ = [
    "ChatChunk",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "DescribeOptions",
    "Embedder",
    "Entity",
    "FilterExpr",
    "GraphStore",
    "GraphStoreCapabilities",
    "LLMCapabilities",
    "LLMProvider",
    "ParseResult",
    "Parser",
    "PluginRegistry",
    "Processor",
    "RawDocument",
    "RerankResult",
    "Reranker",
    "ScoredChunk",
    "SubGraph",
    "TopicSummary",
    "VectorStore",
    "VectorStoreCapabilities",
    "default_plugin",
]
