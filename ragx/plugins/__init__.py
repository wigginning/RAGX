"""Builtin plugin pack. ``register_builtins`` wires every bundled plugin into a
:class:`~ragx.spi.registry.PluginRegistry` without requiring separate packaging
(01-spi.md §1.3: builtin registration is an explicit step of discovery)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ragx.plugins.embed_hash import HashEmbedder
from ragx.plugins.embed_openai_compat import OpenAICompatEmbedder
from ragx.plugins.graph_neo4j import Neo4jGraphStore
from ragx.plugins.graph_nx import NetworkXGraphStore
from ragx.plugins.llm_openai_compat import OpenAICompatibleLLM
from ragx.plugins.parser_asr import ASRParser
from ragx.plugins.parser_av import AVParser
from ragx.plugins.parser_deepdoc import DeepDocParser
from ragx.plugins.parser_mineru import MinerUParser
from ragx.plugins.parser_text import TextMarkdownParser
from ragx.plugins.processor_vlm import VLMDescribeProcessor
from ragx.plugins.vector_es import ESVectorStore
from ragx.plugins.vector_qdrant import QdrantVectorStore
from ragx.plugins.vector_sqlite import SQLiteVectorStore

if TYPE_CHECKING:  # pragma: no cover
    from ragx.spi.registry import PluginRegistry

_BUILTINS: tuple[tuple[str, str, object], ...] = (
    ("parser", "text", TextMarkdownParser),
    ("parser", "deepdoc", DeepDocParser),
    ("parser", "mineru", MinerUParser),
    # experimental (RX-PLG-06): graceful-degrade parsers for audio / video.
    # They construct fine without faster-whisper / PySceneDetect; parse()
    # surfaces UnsupportedFormatError(2002) when the heavy deps are absent.
    ("parser", "asr", ASRParser),
    ("parser", "av", AVParser),
    ("processor", "vlm", VLMDescribeProcessor),
    ("embedder", "hash", HashEmbedder),
    ("embedder", "openai_compat", OpenAICompatEmbedder),
    ("vector_store", "sqlite", SQLiteVectorStore),
    ("vector_store", "es", ESVectorStore),
    ("vector_store", "qdrant", QdrantVectorStore),
    ("graph_store", "nx", NetworkXGraphStore),
    ("graph_store", "neo4j", Neo4jGraphStore),
    ("llm_provider", "openai_compat", OpenAICompatibleLLM),
)


def register_builtins(registry: PluginRegistry) -> list[str]:
    """Register all lite/full bundled plugins; returns the registered names.

    Plugins whose optional third-party deps are missing are registered as
    factories that raise :class:`PluginContractError` on instantiation. The
    registry therefore has a uniform surface; users only hit the error when
    they actually pick that plugin.
    """
    registered: list[str] = []
    for interface, name, cls in _BUILTINS:
        if not registry.has(interface, name):
            registry.register(interface, name, cls)
            registered.append(f"{interface}:{name}")
    return registered


__all__ = ["register_builtins", "LocalFSObjectStore"]

from ragx.plugins.object_store_local import LocalFSObjectStore  # noqa: E402
