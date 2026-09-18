"""Parser contract suite against the built-in text/markdown parser (RX-SPI-02)."""

from __future__ import annotations

from ragx.spi.contracts.parser import ParserContract
from tests.contract.dummy_plugins import DummyParser


class TestTextMarkdownParser(ParserContract):
    async def make_parser(self):
        return DummyParser()
