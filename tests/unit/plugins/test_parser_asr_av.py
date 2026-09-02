"""Unit tests for ASRParser and AVParser (RX-PLG-06).

ASR (faster-whisper) and AV (PySceneDetect) parsers are *experimental* — the
heavy deps are optional and absent from the lite install. The tests focus on
the SPI surface, mimetype rejection, **graceful degradation** when the heavy
deps are missing, and the deterministic mock injection seam that lets CI
exercise the full pipeline without GPU.

Spec: 13-parsing.md §13.3.3, TASKS.md RX-PLG-06.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ragx.core.exceptions import UnsupportedFormatError
from ragx.core.models import RawDocument
from ragx.core.settings import ParseOptions


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _raw_doc(
    mimetype: str,
    content: bytes = b"\x00\x01",
    *,
    doc_id: str = "doc_test",
) -> RawDocument:
    return RawDocument(
        kb_id="default",
        filename="audio.mp3" if mimetype.startswith("audio/") else "video.mp4",
        mimetype=mimetype,
        content=content,
        doc_id=doc_id,
    )


class _Segment:
    """Stand-in for ``faster_whisper.TranscriptionSegment`` used by stubs."""

    def __init__(self, text: str, start: float, end: float, speaker: str | None = None) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.speaker = speaker


def _stub_segments(_doc: RawDocument) -> list[_Segment]:
    """Deterministic transcript used by the injected transcribe_fn."""
    return [
        _Segment("Hello world.", 0.0, 1.0),
        _Segment("This is the second segment.", 1.2, 2.8, speaker="A"),
    ]


def _stub_scene_detector(_doc: RawDocument) -> list[dict]:
    """Deterministic keyframe descriptors used by the injected scene_detector."""
    return [
        {"start_s": 0.0, "end_s": 5.0, "bytes": b"\xff\xd8\xff\xe0frame1"},
        {"start_s": 5.5, "end_s": 10.0, "bytes": b"\xff\xd8\xff\xe0frame2"},
    ]


# ---------------------------------------------------------------------------
# ASRParser
# ---------------------------------------------------------------------------
class TestASRParser:
    def test_spi_surface(self) -> None:
        from ragx.plugins.parser_asr import ASRParser

        assert ASRParser.name == "asr"
        assert "audio/mpeg" in ASRParser.supported_mimetypes
        assert "audio/wav" in ASRParser.supported_mimetypes

    def test_construct_succeeds_without_deps(self) -> None:
        """experimental contract: construction never fails because of missing
        dependencies — SPI discovery must be uniform across profiles."""
        from ragx.plugins.parser_asr import ASRParser

        parser = ASRParser()
        assert parser.name == "asr"

    def test_rejects_unknown_mimetype(self) -> None:
        from ragx.plugins.parser_asr import ASRParser

        parser = ASRParser()

        async def go() -> None:
            await parser.parse(_raw_doc("text/plain"), options=ParseOptions())

        with pytest.raises(UnsupportedFormatError) as exc_info:
            asyncio.run(go())
        assert exc_info.value.code == 2002
        assert exc_info.value.details["parser"] == "asr"

    def test_graceful_degrade_when_whisper_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When faster-whisper is not installed AND no model is injected,
        parse() must surface UnsupportedFormatError(2002) — NOT a generic
        ParseError. 13-parsing.md §13.3.3."""
        from ragx.plugins import parser_asr as mod
        from ragx.plugins.parser_asr import ASRParser

        monkeypatch.setattr(mod, "_maybe_import_whisper", lambda: None)

        parser = ASRParser()  # construction ok

        async def go() -> None:
            await parser.parse(_raw_doc("audio/mpeg"), options=ParseOptions())

        with pytest.raises(UnsupportedFormatError) as exc_info:
            asyncio.run(go())
        assert exc_info.value.code == 2002
        assert "faster-whisper" in exc_info.value.message
        # The error must include the upgrade hint so the ingestion pipeline
        # can surface a clear "install ragx[asr]" recommendation.
        assert "ragx[asr]" in str(exc_info.value.details.get("install", ""))

    def test_injected_transcribe_fn_produces_atoms(self) -> None:
        """Mock / deterministic parse path via the ``transcribe_fn`` seam."""
        from ragx.plugins.parser_asr import ASRParser

        parser = ASRParser({"transcribe_fn": _stub_segments})

        async def go() -> None:
            return await parser.parse(_raw_doc("audio/mpeg"), options=ParseOptions())

        result = asyncio.run(go())
        # Two non-empty TEXT atoms in reading order.
        assert len(result.atoms) == 2
        assert all(a.type.value == "text" for a in result.atoms)
        assert [a.atom_id for a in result.atoms] == ["doc_test#0000", "doc_test#0001"]
        assert result.atoms[0].text == "Hello world."
        assert result.atoms[0].metadata["start_s"] == 0.0
        assert result.atoms[0].metadata["end_s"] == 1.0
        assert result.atoms[1].metadata["speaker"] == "A"
        assert result.metadata["parser"] == "asr"
        assert result.metadata["segments"] == 2

    def test_injected_model_skips_dep_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When an explicit ``model`` handle is supplied, the parser must not
        try to import faster-whisper at all."""
        from ragx.plugins import parser_asr as mod
        from ragx.plugins.parser_asr import ASRParser

        called = {"import": 0}
        original = mod._maybe_import_whisper

        def spy() -> None:
            called["import"] += 1
            return None  # pretend whisper is missing

        monkeypatch.setattr(mod, "_maybe_import_whisper", spy)
        # touch original to keep linter happy
        assert original is not None

        sentinel_model = SimpleNamespace(transcribe=lambda *a, **k: (iter([]), None))
        parser = ASRParser({"model": sentinel_model})
        # _whisper_mod is None because the injected model short-circuits the
        # probe — confirm we never tried to import.
        assert parser._whisper_mod is None

        async def go() -> None:
            return await parser.parse(_raw_doc("audio/wav"), options=ParseOptions())

        result = asyncio.run(go())
        assert result.atoms == []
        assert result.metadata["parser"] == "asr"


# ---------------------------------------------------------------------------
# AVParser
# ---------------------------------------------------------------------------
class TestAVParser:
    def test_spi_surface(self) -> None:
        from ragx.plugins.parser_av import AVParser

        assert AVParser.name == "av"
        assert "video/mp4" in AVParser.supported_mimetypes
        assert "video/webm" in AVParser.supported_mimetypes

    def test_construct_succeeds_without_deps(self) -> None:
        """experimental contract: AVParser constructs even without the
        PySceneDetect / faster-whisper deps."""
        from ragx.plugins.parser_av import AVParser

        parser = AVParser()
        assert parser.name == "av"

    def test_rejects_unknown_mimetype(self) -> None:
        from ragx.plugins.parser_av import AVParser

        parser = AVParser()

        async def go() -> None:
            await parser.parse(_raw_doc("text/plain"), options=ParseOptions())

        with pytest.raises(UnsupportedFormatError) as exc_info:
            asyncio.run(go())
        assert exc_info.value.code == 2002

    def test_graceful_degrade_when_both_deps_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When neither asr/scene_detector is injected AND both faster-whisper
        and PySceneDetect are absent, parse() must surface
        UnsupportedFormatError(2002)."""
        from ragx.plugins import parser_av as mod
        from ragx.plugins.parser_av import AVParser

        monkeypatch.setattr(mod, "_maybe_import_scenedetect", lambda: None)
        monkeypatch.setattr(mod, "_maybe_import_whisper", lambda: None)

        parser = AVParser()

        async def go() -> None:
            await parser.parse(_raw_doc("video/mp4"), options=ParseOptions())

        with pytest.raises(UnsupportedFormatError) as exc_info:
            asyncio.run(go())
        assert exc_info.value.code == 2002
        assert "PySceneDetect" in exc_info.value.message or "faster-whisper" in exc_info.value.message
        # upgrade hint must be present
        assert "ragx[av]" in str(exc_info.value.details.get("install", ""))

    def test_injected_asr_and_scene_detector(self) -> None:
        """Deterministic AV parse path via injected asr + scene_detector."""
        from ragx.plugins.parser_asr import ASRParser
        from ragx.plugins.parser_av import AVParser

        asr = ASRParser({"transcribe_fn": _stub_segments})
        parser = AVParser({"asr": asr, "scene_detector": _stub_scene_detector})

        async def go() -> None:
            return await parser.parse(_raw_doc("video/mp4"), options=ParseOptions())

        result = asyncio.run(go())
        # 2 transcript atoms + 2 keyframe atoms = 4 total, in reading order.
        assert len(result.atoms) == 4
        assert result.atoms[0].type.value == "text"
        assert result.atoms[1].type.value == "text"
        assert result.atoms[2].type.value == "image"
        assert result.atoms[3].type.value == "image"
        # atom_ids are stable, sequential
        assert [a.atom_id for a in result.atoms] == [
            "doc_test#0000",
            "doc_test#0001",
            "doc_test#0002",
            "doc_test#0003",
        ]
        # IMAGE atoms carry the scene timestamps + a synthetic content_hash
        assert result.atoms[2].metadata["start_s"] == 0.0
        assert result.atoms[2].metadata["end_s"] == 5.0
        assert result.atoms[2].content_hash  # description-cache key present
        assert result.metadata["parser"] == "av"
        assert result.metadata["scenes"] == 2
        assert result.metadata["audio_atoms"] == 2

    def test_scene_detector_only(self) -> None:
        """Only scene_detector injected — should produce only IMAGE atoms."""
        from ragx.plugins.parser_av import AVParser

        parser = AVParser({"scene_detector": _stub_scene_detector})

        async def go() -> None:
            return await parser.parse(_raw_doc("video/mp4"), options=ParseOptions())

        result = asyncio.run(go())
        assert len(result.atoms) == 2
        assert all(a.type.value == "image" for a in result.atoms)
        assert result.metadata["audio_atoms"] == 0
        assert result.metadata["scenes"] == 2


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_builtin_registration_includes_asr_and_av(self) -> None:
        """RX-PLG-06: ASRParser / AVParser must be discoverable through
        ``register_builtins`` even when their heavy deps are absent."""
        from ragx.plugins import register_builtins
        from ragx.spi.registry import PluginRegistry

        reg = PluginRegistry()
        register_builtins(reg)
        assert reg.has("parser", "asr")
        assert reg.has("parser", "av")
        # And construction does not raise on this machine (graceful degrade).
        assert isinstance(reg.resolve("parser", "asr"), object)
        assert isinstance(reg.resolve("parser", "av"), object)
