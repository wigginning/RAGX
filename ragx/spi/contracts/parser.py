"""Parser contract (01-spi.md §1.4)."""

from __future__ import annotations

import pytest

from ragx.core.exceptions import ParseError, UnsupportedFormatError
from ragx.core.models import Atom, AtomType, RawDocument
from ragx.core.settings import ParseOptions
from ragx.spi.contracts.base import SAMPLE_MARKDOWN, ContractBase
from ragx.spi.interfaces import Parser, ParseResult


class ParserContract(ContractBase):
    """Inherit and implement :meth:`make_parser` to inherit this whole suite."""

    async def make(self) -> Parser:
        return await self.make_parser()

    async def make_parser(self) -> Parser:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def _doc(self, content: str, mimetype: str = "text/markdown") -> RawDocument:
        return RawDocument(
            doc_id="doc_test000000000000000000000001",
            kb_id="kb_test",
            filename="sample.md",
            mimetype=mimetype,
            content=content,
        )

    # -- suite -------------------------------------------------------------
    @pytest.mark.contract
    async def test_supported_mimetypes_declared(self) -> None:
        parser = await self._get()
        assert isinstance(parser.supported_mimetypes, list)
        assert parser.supported_mimetypes, "parser must declare at least one mimetype"
        assert isinstance(parser.name, str) and parser.name

    @pytest.mark.contract
    async def test_parse_returns_atoms(self) -> None:
        parser = await self._get()
        result = await parser.parse(self._doc(SAMPLE_MARKDOWN), options=ParseOptions())
        assert isinstance(result, ParseResult)
        assert result.atoms, "a non-empty markdown sample must yield at least one atom"
        assert all(isinstance(a, Atom) for a in result.atoms)

    @pytest.mark.contract
    async def test_atom_ids_are_stable_and_positional(self) -> None:
        """atom_id == f'{doc_id}#{seq:04d}' and identical across reruns (01-spi §1.2)."""
        parser = await self._get()
        doc = self._doc(SAMPLE_MARKDOWN)
        first = await parser.parse(doc, options=ParseOptions())
        second = await parser.parse(doc, options=ParseOptions())
        assert [a.atom_id for a in first.atoms] == [a.atom_id for a in second.atoms]
        for index, atom in enumerate(first.atoms):
            assert atom.atom_id == f"{doc.doc_id}#{index:04d}", atom.atom_id
            assert atom.doc_id == doc.doc_id

    @pytest.mark.contract
    async def test_atoms_carry_type_and_content_hash(self) -> None:
        parser = await self._get()
        result = await parser.parse(self._doc(SAMPLE_MARKDOWN), options=ParseOptions())
        for atom in result.atoms:
            assert isinstance(atom.type, AtomType)
            assert atom.content_hash, "content_hash is the description-cache key (03 §3.4.1)"
            if atom.type == AtomType.TEXT:
                assert (atom.text or "").strip()

    @pytest.mark.contract
    async def test_content_hash_is_deterministic(self) -> None:
        parser = await self._get()
        doc = self._doc(SAMPLE_MARKDOWN)
        a = await parser.parse(doc, options=ParseOptions())
        b = await parser.parse(doc, options=ParseOptions())
        assert [x.content_hash for x in a.atoms] == [x.content_hash for x in b.atoms]

    @pytest.mark.contract
    async def test_text_content_yields_text_atoms(self) -> None:
        parser = await self._get()
        if "text/plain" not in parser.supported_mimetypes:
            pytest.skip("parser does not support text/plain")
        result = await parser.parse(
            self._doc("第一行。\n\n第二段，换行分隔。", "text/plain"), options=ParseOptions()
        )
        assert result.atoms
        assert any(a.type == AtomType.TEXT for a in result.atoms)

    @pytest.mark.contract
    async def test_empty_input_yields_no_atoms(self) -> None:
        parser = await self._get()
        result = await parser.parse(self._doc("", "text/plain"), options=ParseOptions())
        assert result.atoms == []

    @pytest.mark.contract
    async def test_unsupported_mimetype_rejected(self) -> None:
        """Unlisted mimetype must raise a domain error, never a raw exception."""
        parser = await self._get()
        probes = (
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        unsupported = [m for m in probes if m not in parser.supported_mimetypes]
        if not unsupported:
            pytest.skip("parser claims support for every probed format")
        with pytest.raises((ParseError, UnsupportedFormatError)):
            await parser.parse(
                self._doc("\x00\x01\x02", unsupported[0]), options=ParseOptions()
            )
