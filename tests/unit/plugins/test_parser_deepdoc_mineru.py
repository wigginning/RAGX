"""Unit tests for DeepDocParser and MinerUParser (RX-PLG-05).

The actual heavy dependencies (pdfplumber, pymupdf, pytesseract) are not
installed in the lite profile. The tests focus on:

* SPI compliance — name, supported_mimetypes, async interface
* Mimetype rejection — unknown mimetype → UnsupportedFormatError(2002)
* PluginContractError on construction when the optional deps are missing
"""

from __future__ import annotations

import pytest

from ragx.core.exceptions import (
    PluginContractError,
    UnsupportedFormatError,
)
from ragx.core.models import RawDocument
from ragx.core.settings import ParseOptions


def _raw_doc(mimetype: str, content: bytes = b"x") -> RawDocument:
    return RawDocument(
        kb_id="default",
        filename="doc.bin",
        mimetype=mimetype,
        content=content,
        doc_id="doc_test",
    )


class TestDeepDocParser:
    def test_spi_surface(self) -> None:
        from ragx.plugins.parser_deepdoc import DeepDocParser

        assert DeepDocParser.name == "deepdoc"
        assert "application/pdf" in DeepDocParser.supported_mimetypes
        assert "image/png" in DeepDocParser.supported_mimetypes

    def test_rejects_unknown_mimetype(self) -> None:
        from ragx.plugins.parser_deepdoc import DeepDocParser

        # Construction may fail when pdfplumber/pypdf are missing — guard.
        try:
            parser = DeepDocParser()
        except PluginContractError:
            pytest.skip("pdfplumber/pypdf not installed")

        async def go() -> None:
            await parser.parse(_raw_doc("text/plain"), options=ParseOptions())

        import asyncio

        with pytest.raises(UnsupportedFormatError):
            asyncio.run(go())

    def test_image_mimetype_accepted(self) -> None:
        from ragx.plugins.parser_deepdoc import DeepDocParser

        try:
            parser = DeepDocParser()
        except PluginContractError:
            pytest.skip("pdfplumber/pypdf not installed")

        assert "image/jpeg" in parser.supported_mimetypes

    def test_constructor_raises_without_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When BOTH pdfplumber and pypdf are missing the parser must refuse
        construction with a PluginContractError."""
        from ragx.plugins import parser_deepdoc as mod

        monkeypatch.setattr(mod, "_maybe_import_pdfplumber", lambda: None)
        monkeypatch.setattr(mod, "_maybe_import_pypdf", lambda: None)
        with pytest.raises(PluginContractError):
            mod.DeepDocParser()


class TestMinerUParser:
    def test_spi_surface(self) -> None:
        from ragx.plugins.parser_mineru import MinerUParser

        assert MinerUParser.name == "mineru"
        assert "application/pdf" in MinerUParser.supported_mimetypes

    def test_constructor_requires_pymupdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ragx.plugins import parser_mineru as mod

        monkeypatch.setattr(mod, "_maybe_import_pymupdf", lambda: None)
        with pytest.raises(PluginContractError):
            mod.MinerUParser()

    def test_rejects_unknown_mimetype(self) -> None:
        from ragx.plugins.parser_mineru import MinerUParser

        try:
            parser = MinerUParser()
        except PluginContractError:
            pytest.skip("pymupdf not installed")

        import asyncio

        async def go() -> None:
            await parser.parse(_raw_doc("text/plain"), options=ParseOptions())

        with pytest.raises(UnsupportedFormatError):
            asyncio.run(go())

    def test_formula_token_recognition(self) -> None:
        """The formula heuristic must recognise a few common LaTeX patterns."""
        from ragx.plugins.parser_mineru import _FORMULA_TOKEN

        assert _FORMULA_TOKEN.match("$$x = y + z$$")
        assert _FORMULA_TOKEN.match("\\frac{1}{2}")
        assert _FORMULA_TOKEN.match("\\sum_{i=1}^n i")
        assert _FORMULA_TOKEN.match("\\int_a^b x dx")
        # and not match plain text
        assert not _FORMULA_TOKEN.match("Hello world")
        assert not _FORMULA_TOKEN.match("RAGX is great")


class TestRegistration:
    def test_builtin_registration_includes_deepdoc_and_mineru(self) -> None:
        from ragx.plugins import register_builtins
        from ragx.spi.registry import PluginRegistry

        reg = PluginRegistry()
        register_builtins(reg)
        # Registration always succeeds; construction may fail when deps are
        # missing — that's by design.
        assert reg.has("parser", "deepdoc")
        assert reg.has("parser", "mineru")
        assert reg.has("parser", "text")
