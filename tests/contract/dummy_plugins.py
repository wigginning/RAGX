"""In-memory dummy plugin implementations for the contract suites (RX-SPI-02).

Each dummy is a *minimal but correct* implementation of one SPI protocol. The
contract suites (``ragx.spi.contracts.*``) are inherited by a test class that
implements a single ``make_*`` factory returning one of these dummies; the
whole suite then runs against it. This proves the contract-suite pattern
(01-spi.md §1.4): a plugin that passes the shared suite is SPI-conformant.

All dummies are deterministic and never touch the network.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from pydantic import BaseModel

from ragx.core.models import (
    Chunk,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Parser - reuse the built-in text/markdown parser (already verified).
# ---------------------------------------------------------------------------
from ragx.plugins.parser_text import TextMarkdownParser  # noqa: E402
from ragx.spi.interfaces import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    LLMCapabilities,
    RerankResult,
)


class DummyParser(TextMarkdownParser):
    """The built-in text/markdown parser is a conformant Parser."""


# ---------------------------------------------------------------------------
# Processor - VLMDescribeProcessor wired to a deterministic fake chat callable.
# ---------------------------------------------------------------------------
from ragx.plugins.processor_vlm import VLMDescribeProcessor  # noqa: E402


async def _fake_chat(messages: list[tuple[str, str]]) -> str:
    """Deterministic, idempotent description with a trailing confidence line."""
    user = messages[-1][1] if messages else ""
    return f"这是一段由 dummy 处理器生成的确定性描述。内容：{user[:40]}。confidence: 0.85"


class DummyProcessor(VLMDescribeProcessor):
    def __init__(self) -> None:
        super().__init__({"chat": _fake_chat, "model": "dummy-vlm"})


# ---------------------------------------------------------------------------
# Embedder - reuse the built-in hash embedder (deterministic, zero-dep).
# ---------------------------------------------------------------------------
from ragx.plugins.embed_hash import HashEmbedder  # noqa: E402


class DummyEmbedder(HashEmbedder):
    """The built-in hash embedder is a conformant Embedder."""


# ---------------------------------------------------------------------------
# Reranker - a deterministic keyword-overlap scorer (no external model).
# ---------------------------------------------------------------------------
class DummyReranker:
    """Scores chunks by query-token overlap, normalised to [0, 1]."""

    name: str = "dummy-overlap"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in text.lower().split() if t}

    async def rerank(
        self, query: str, chunks: list[Chunk], *, top_k: int
    ) -> list[RerankResult]:
        if not chunks:
            return []
        q = self._tokens(query)
        scored: list[tuple[float, str]] = []
        for chunk in chunks:
            c = self._tokens(chunk.text)
            overlap = len(q & c)
            score = overlap / max(len(q), 1) if q else 0.0
            scored.append((score, chunk.chunk_id))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RerankResult(chunk_id=cid, score=score) for score, cid in scored[: max(top_k, 0)]
        ]


# ---------------------------------------------------------------------------
# VectorStore - reuse the built-in SQLite store (dim=8, kb-scoped).
# ---------------------------------------------------------------------------
from ragx.plugins.vector_sqlite import SQLiteVectorStore  # noqa: E402


class DummyVectorStore(SQLiteVectorStore):
    """SQLite store configured for the contract kb at dim=8."""

    def __init__(self) -> None:
        super().__init__({"kb_id": "kb_contract", "dim": 8})


# ---------------------------------------------------------------------------
# GraphStore - reuse the built-in NetworkX store (kb-scoped).
# ---------------------------------------------------------------------------
from ragx.plugins.graph_nx import NetworkXGraphStore  # noqa: E402


class DummyGraphStore(NetworkXGraphStore):
    def __init__(self) -> None:
        super().__init__({"kb_id": "kb_contract"})


# ---------------------------------------------------------------------------
# LLMProvider - a mock transport: deterministic text / JSON, no network.
# ---------------------------------------------------------------------------
class DummyLLMProvider:
    """A conformant LLMProvider whose transport is a pure function of the prompt."""

    name: str = "dummy"
    capabilities: LLMCapabilities = LLMCapabilities(
        supports_json_mode=True, supports_stream=True
    )

    @staticmethod
    def _answer(req: ChatRequest) -> str:
        if req.json_mode:
            return '{"type": "answer", "tasks": ["t1", "t2"]}'
        return "RAGX 的分层架构由 SPI 插件层、核心层与检索/摄入/LLM 等能力层组成。"

    async def chat(self, req: ChatRequest) -> ChatResponse:
        text = self._answer(req)
        return ChatResponse(
            text=text,
            usage=TokenUsage(prompt_tokens=12, completion_tokens=8, total=20),
            model=req.model or self.name,
            raw={"dummy": True},
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        text = self._answer(req)
        for i in range(0, len(text), 4):
            yield ChatChunk(delta=text[i : i + 4])
        yield ChatChunk(delta="", finish_reason="stop")

    async def structured(
        self, req: ChatRequest, *, schema: type[BaseModel]
    ) -> BaseModel:
        text = self._answer(req)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"type": "answer", "tasks": []}
        return schema(**data)

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None
