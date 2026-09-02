"""Processor contract suite against the VLM processor with a fake chat (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.processor import ProcessorContract
from tests.contract.dummy_plugins import DummyProcessor


class TestVLMProcessor(ProcessorContract):
    async def make_processor(self):
        return DummyProcessor()
