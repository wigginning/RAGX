"""ASRParser (11-plugins-builtin.md §11.2.7, RX-PLG-06, experimental).

Audio transcription parser using ``faster-whisper``. Production deployments
install ``faster-whisper`` (via the ``ragx[asr]`` extra) and configure a
model size (tiny / base / small / medium / large-v3) via ``KBConfig.parsing``.

experimental mode contract
--------------------------

This parser is **always** constructable: construction never fails because of
missing dependencies, so the SPI discovery path can register and resolve it
uniformly. When ``faster-whisper`` is not installed AND no stub ``model`` is
injected via the config, ``parse()`` raises :class:`UnsupportedFormatError`
(code **2002**) — the documented graceful-degrade signal for an
unavailable dependency (00-overview.md §0.3, 13-parsing.md §13.3.3).

Tests inject a stub ``model`` (or a stub ``transcribe_fn``) via
``__init__`` to bypass the optional dep and exercise the SPI contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ragx.core.exceptions import (
    ParseError,
    UnsupportedFormatError,
)
from ragx.core.hashing import compute_atom_content_hash
from ragx.core.models import Atom, AtomType
from ragx.core.settings import ParseOptions
from ragx.spi.interfaces import ParseResult

logger = logging.getLogger("ragx.plugins.parser_asr")

_NAME = "asr"
_SUPPORTED_MIMETYPES: list[str] = [
    "audio/mpeg",         # mp3
    "audio/wav",
    "audio/x-wav",
    "audio/x-m4a",
    "audio/mp4",
    "audio/ogg",
]


def _maybe_import_whisper() -> Any:
    try:
        import faster_whisper  # type: ignore[import-not-found]

        return faster_whisper
    except ImportError:
        return None


def _split_paragraphs(text: str) -> list[str]:
    """Coarse paragraph split on long pauses (the unit ``faster-whisper``
    exposes via ``word_timestamps``). This lite fallback splits on
    punctuation + double newlines."""
    import re

    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


class ASRParser:
    """Audio transcription parser (faster-whisper backend, experimental)."""

    name: str = _NAME
    supported_mimetypes: list[str] = list(_SUPPORTED_MIMETYPES)

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.model_size: str = str(self.config.get("model_size", "small"))
        self.language: str | None = self.config.get("language")
        self.beam_size: int = int(self.config.get("beam_size", 5))

        # Test seams — callers (and tests) can inject an explicit handle so the
        # parser can run without the heavier faster-whisper install. Both are
        # optional; when both are None we lazy-import faster-whisper on first
        # parse() and, if still missing, surface UnsupportedFormatError(2002).
        self.model: Any = self.config.get("model")
        self.transcribe_fn: Callable[..., Any] | None = self.config.get("transcribe_fn")

        # Track whether the optional dep is available; we don't probe at
        # __init__ time so the SPI registry can build this parser uniformly
        # regardless of which extras are installed.
        self._whisper_mod = _maybe_import_whisper() if (self.model is None and self.transcribe_fn is None) else None

    # -- SPI ---------------------------------------------------------------
    async def startup(self) -> None:  # pragma: no cover - trivial hook
        return None

    async def shutdown(self) -> None:  # pragma: no cover - trivial hook
        return None

    async def parse(
        self, doc: Any, *, options: ParseOptions
    ) -> ParseResult:  # noqa: ANN001
        mt = doc.mimetype
        if mt not in self.supported_mimetypes:
            raise UnsupportedFormatError(
                "mimetype not supported by the asr parser",
                code=2002,
                details={
                    "mimetype": mt,
                    "supported": self.supported_mimetypes,
                    "parser": self.name,
                },
            )

        # Graceful degrade (13-parsing.md §13.3.3, experimental): if no model
        # is injected and faster-whisper isn't installed, surface a clear
        # unsupported error (2002) so the ingestion pipeline can emit the
        # "install ragx[asr]" hint to the user.
        if self.model is None and self.transcribe_fn is None and self._whisper_mod is None:
            raise UnsupportedFormatError(
                "faster-whisper is not installed; ASRParser is unavailable",
                code=2002,
                details={
                    "parser": self.name,
                    "mimetype": mt,
                    "install": "pip install 'ragx[asr]'",
                },
            )

        try:
            return self._transcribe(doc)
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise ParseError(
                "asr parse failed",
                code=2001,
                details={"mimetype": mt, "error": str(exc)},
            ) from exc

    # -- internal -----------------------------------------------------------
    def _ensure_model(self) -> Any:
        if self.model is not None:
            return self.model
        if self._whisper_mod is None:
            # Should be unreachable — guarded in parse() above. Defensive.
            raise UnsupportedFormatError(
                "faster-whisper is not installed",
                code=2002,
                details={"parser": self.name},
            )
        # WhisperModel is the canonical constructor.
        self.model = self._whisper_mod.WhisperModel(self.model_size)
        return self.model

    def _transcribe(self, doc: Any) -> ParseResult:
        """Run the ASR model on the audio bytes and emit TEXT atoms."""
        # The injected transcribe_fn wins over the real model — used by tests
        # to drive deterministic output.
        if self.transcribe_fn is not None:
            return self._segments_to_atoms(doc, list(self.transcribe_fn(doc)))

        model = self._ensure_model()
        # faster-whisper exposes transcribe(audio, ...) returning
        # (segments_iterable, info). We write the audio to a temp file-like
        # in production; here we pass raw bytes via a NamedTemporaryFile.
        import os
        import tempfile

        content = doc.content if isinstance(doc.content, bytes) else doc.content.encode("utf-8")
        with tempfile.NamedTemporaryFile(suffix=self._suffix(doc), delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            segments_iter, _info = model.transcribe(
                tmp_path, beam_size=self.beam_size, language=self.language,
                word_timestamps=True,
            )
            segments = list(segments_iter)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return self._segments_to_atoms(doc, segments)

    def _segments_to_atoms(self, doc: Any, segments: list[Any]) -> ParseResult:
        """Translate a list of segments (with ``start/end/text`` attrs) into
        TEXT atoms. Public for testing/determinism."""
        atoms: list[Atom] = []
        for seq, seg in enumerate(segments):
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            meta = {
                "start_s": float(getattr(seg, "start", 0.0)),
                "end_s": float(getattr(seg, "end", 0.0)),
                "speaker": getattr(seg, "speaker", None),
            }
            atoms.append(
                Atom(
                    atom_id=f"{doc.doc_id}#{seq:04d}",
                    doc_id=doc.doc_id or "",
                    type=AtomType.TEXT,
                    text=text,
                    content_hash=compute_atom_content_hash(text),
                    page=None,
                    metadata=meta,
                )
            )
        return ParseResult(atoms=atoms, metadata={"parser": self.name, "segments": len(atoms)})

    @staticmethod
    def _suffix(doc: Any) -> str:
        mt = (doc.mimetype or "").lower()
        if "mpeg" in mt:
            return ".mp3"
        if "wav" in mt:
            return ".wav"
        if "m4a" in mt or "mp4" in mt:
            return ".m4a"
        if "ogg" in mt:
            return ".ogg"
        return ".audio"


__all__ = ["ASRParser"]
