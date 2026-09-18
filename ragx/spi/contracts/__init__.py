"""Contract test base classes (01-spi.md §1.4).

Each base class is an abstract pytest fixture object: a plugin author inherits
it, implements the single ``make_*()`` factory, and inherits the whole suite::

    class TestSQLiteVectorStore(VectorStoreContract):
        async def make_store(self) -> VectorStore:
            return SQLiteVectorStore({"path": ":memory:", "dim": 16})

Capabilities drive skips: a suite never asserts behaviour a plugin declared it
does not provide (``supports_bm25`` / ``supports_community`` /
``supports_stream`` / ``supports_json_mode``).
"""

from ragx.spi.contracts.base import ContractBase
from ragx.spi.contracts.embedder import EmbedderContract
from ragx.spi.contracts.graph_store import GraphStoreContract
from ragx.spi.contracts.llm_provider import LLMProviderContract
from ragx.spi.contracts.parser import ParserContract
from ragx.spi.contracts.processor import ProcessorContract
from ragx.spi.contracts.reranker import RerankerContract
from ragx.spi.contracts.vector_store import VectorStoreContract

__all__ = [
    "ContractBase",
    "EmbedderContract",
    "GraphStoreContract",
    "LLMProviderContract",
    "ParserContract",
    "ProcessorContract",
    "RerankerContract",
    "VectorStoreContract",
]
