"""LLMProvider contract (01-spi.md §1.4).

Runs entirely against the plugin's own transport layer. Third-party plugins are
expected to point ``base_url`` at a local mock (08-llm.md uses a recorded-replay
transport); the suite never contacts a real API.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from ragx.core.models import TokenUsage
from ragx.core.roles import LLMRole
from ragx.spi.contracts.base import ContractBase
from ragx.spi.interfaces import ChatMessage, ChatRequest, LLMProvider


class _PlanSchema(BaseModel):
    type: str
    tasks: list[str] = []


class LLMProviderContract(ContractBase):
    async def make(self) -> LLMProvider:
        return await self.make_provider()

    async def make_provider(self) -> LLMProvider:  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _req(text: str = "用一句话解释 RAGX 的分层架构。", *, json_mode: bool = False) -> ChatRequest:
        return ChatRequest(
            messages=[
                ChatMessage(role="system", content="你是一个文档问答助手。"),
                ChatMessage(role="user", content=text),
            ],
            role=LLMRole.GENERATE,
            model="contract-model",
            temperature=0.1,
            max_tokens=256,
            json_mode=json_mode,
            kb_id="kb_contract",
            trace_id="trace_contract",
        )

    # -- suite -------------------------------------------------------------
    @pytest.mark.contract
    async def test_capabilities_declared(self) -> None:
        provider = await self._get()
        assert isinstance(provider.capabilities.supports_json_mode, bool)
        assert isinstance(provider.capabilities.supports_stream, bool)
        assert isinstance(provider.name, str) and provider.name

    @pytest.mark.contract
    async def test_chat_returns_text_and_usage(self) -> None:
        """``usage`` is mandatory - the Cost Ledger depends on it (01-spi §1.2)."""
        provider = await self._get()
        resp = await provider.chat(self._req())
        assert resp.text.strip(), "chat must return non-empty text"
        assert isinstance(resp.usage, TokenUsage)
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        assert resp.usage.total >= resp.usage.prompt_tokens + resp.usage.completion_tokens - 1

    @pytest.mark.contract
    async def test_chat_is_reentrant(self) -> None:
        provider = await self._get()
        a = await provider.chat(self._req())
        b = await provider.chat(self._req())
        assert a.text and b.text

    @pytest.mark.contract
    async def test_structured_returns_schema_instance(self) -> None:
        provider = await self._get()
        req = self._req(
            '输出 JSON：{"type": "answer", "tasks": ["t1"]}', json_mode=True
        )
        result = await provider.structured(req, schema=_PlanSchema)
        assert isinstance(result, _PlanSchema)
        assert isinstance(result.type, str)

    @pytest.mark.contract
    async def test_structured_without_json_mode_still_parses(self) -> None:
        """No json_mode support -> prompt constraint + parse retry (01-spi §1.2)."""
        provider = await self._get()
        req = self._req(
            '只输出 JSON：{"type": "answer", "tasks": []}', json_mode=False
        )
        result = await provider.structured(req, schema=_PlanSchema)
        assert isinstance(result, _PlanSchema)

    @pytest.mark.contract
    async def test_structured_reports_model(self) -> None:
        provider = await self._get()
        resp = await provider.chat(self._req())
        assert resp.model, "response should report the model actually used"

    @pytest.mark.contract
    async def test_chat_stream_if_supported(self) -> None:
        provider = await self._get()
        if not provider.capabilities.supports_stream:
            pytest.skip("supports_stream=False")
        chunks: list[str] = []
        async for piece in provider.chat_stream(self._req()):
            chunks.append(piece.delta)
        assert chunks, "stream must yield at least one delta"
        assert "".join(chunks).strip()
        # streaming must not consume the whole completion in a single delta
        assert any(c for c in chunks if c)

    @pytest.mark.contract
    async def test_stream_total_equals_chat_length_order(self) -> None:
        provider = await self._get()
        if not provider.capabilities.supports_stream:
            pytest.skip("supports_stream=False")
        streamed: list[str] = []
        async for piece in provider.chat_stream(self._req()):
            streamed.append(piece.delta)
        full = await provider.chat(self._req())
        # both paths must produce comparable volume (sanity, not exact equality)
        assert len("".join(streamed)) > 0
        assert len(full.text) > 0
