"""LLMProvider contract suite against the mock transport (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.llm_provider import LLMProviderContract
from tests.contract.dummy_plugins import DummyLLMProvider


class TestDummyLLMProvider(LLMProviderContract):
    async def make_provider(self):
        return DummyLLMProvider()
