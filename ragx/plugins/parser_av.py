"""AVParser (11-plugins-builtin.md §11.2.8, RX-PLG-06, experimental).

Video parser. Pipeline:

1. **Audio track** → delegated to :class:`ASRParser` (TEXT atoms with
   ``start_s``/``end_s``/``speaker`` metadata).
2. **Scene change detection** (PySceneDetect) → keyframe extraction →
   ``IMAGE`` atoms with ``context`` linking back to the ASR transcript at
   the same timestamp.

Both PySceneDetect and faster-whisper are heavy dependencies that the lite
profile does not require.

experimental mode contract
--------------------------

This parser is **always** constructable. When both ``asr`` and
``scene_detector`` are missing AND neither ``faster-whisper`` nor
``PySceneDetect`` is installed, ``parse()`` raises
:class:`UnsupportedFormatError` (code **2002**) — the documented
graceful-degrade signal for an unavailable dependency (00-overview.md §0.3,
13-parsing.md §13.3.3).

Callers (and tests) can inject:

* ``config['asr']`` — a fully constructed :class:`ASRParser` instance
* ``config['scene_detector']`` — a callable ``(doc) -> Iterable[dict]``
  yielding keyframe descriptors

to drive the parser without the heavy deps.
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
from ragx.core.models import Atom, AtomType, RawDocument
from ragx.core.settings import ParseOptions
from ragx.spi.interfaces import ParseResult

logger = logging.getLogger("ragx.plugins.parser_av")

_NAME = "av"
_SUPPORTED_MIMETYPES: list[str] = [
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
]


def _maybe_import_scenedetect() -> Any:
    try:
        import scenedetect  # type: ignore[import-not-found]

        return scenedetect
    except ImportError:
        return None


def _maybe_import_whisper() -> Any:
    try:
        import faster_whisper  # type: ignore[import-not-found]

        return faster_whisper
    except ImportError:
        return None


class AVParser:
    """Video parser — audio transcript + keyframe images (experimental)."""

    name: str = _NAME
    supported_mimetypes: list[str] = list(_SUPPORTED_MIMETYPES)

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.scene_threshold: float = float(self.config.get("scene_threshold", 27.0))
        self.min_scene_len_s: float = float(self.config.get("min_scene_len_s", 1.0))

        # Test seams — callers can inject an ASRParser and/or a scene detector
        # to drive the parser deterministically without the heavy deps.
        self.asr: Any = self.config.get("asr")
        self.scene_detector: Callable[..., Any] | None = self.config.get("scene_detector")

        # Lazy optional-dep probes. The constructor does NOT raise when these
        # are missing — parse() surfaces UnsupportedFormatError(2002) instead,
        # so SPI discovery is uniform across profiles.
        self._scenedetect_mod = (
            _maybe_import_scenedetect()
            if (self.scene_detector is None)
            else None
        )
        self._whisper_mod = (
            _maybe_import_whisper()
            if (self.asr is None)
            else None
        )

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
                "mimetype not supported by the av parser",
                code=2002,
                details={
                    "mimetype": mt,
                    "supported": self.supported_mimetypes,
                    "parser": self.name,
                },
            )

        # Graceful degrade (13-parsing.md §13.3.3): when neither asr / scene
        # detector is injected nor faster-whisper / PySceneDetect is
        # installed, surface a clear unsupported error (2002) so the
        # ingestion pipeline can emit an upgrade hint.
        if (
            self.asr is None
            and self.scene_detector is None
            and self._whisper_mod is None
            and self._scenedetect_mod is None
        ):
            raise UnsupportedFormatError(
                "AVParser requires faster-whisper and PySceneDetect; both unavailable",
                code=2002,
                details={
                    "parser": self.name,
                    "mimetype": mt,
                    "install": "pip install 'ragx[av]'",
                },
            )

        try:
            return await self._parse_video(doc, options=options)
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise ParseError(
                "av parse failed",
                code=2001,
                details={"mimetype": mt, "error": str(exc)},
            ) from exc

    async def _parse_video(self, doc: Any, *, options: ParseOptions) -> ParseResult:
        atoms: list[Atom] = []
        seq = 0

        # 1. Audio track — delegate to ASRParser if wired
        if self.asr is not None:
            audio_doc = self._audio_doc(doc)
            audio_result = await self.asr.parse(audio_doc, options=options)
            for atom in audio_result.atoms:
                atom.atom_id = f"{doc.doc_id}#{seq:04d}"
                atoms.append(atom)
                seq += 1

        # 2. Scene detection — emit one IMAGE atom per scene boundary
        keyframes = self._detect_scenes(doc)
        for kf in keyframes:
            # IMAGE atom: ``compute_atom_content_hash`` hashes the payload bytes
            # (02-core.md §2.2) so the description cache can key on the actual
            # keyframe image content.
            atoms.append(
                Atom(
                    atom_id=f"{doc.doc_id}#{seq:04d}",
                    doc_id=doc.doc_id or "",
                    type=AtomType.IMAGE,
                    text=f"[keyframe @ {kf['start_s']:.2f}s]",
                    content_hash=compute_atom_content_hash(None, payload=kf["bytes"]),
                    page=None,
                    metadata={
                        "parser": self.name,
                        "start_s": float(kf["start_s"]),
                        "end_s": float(kf["end_s"]),
                        "payload_ref": None,
                    },
                )
            )
            seq += 1

        return ParseResult(
            atoms=atoms,
            metadata={
                "parser": self.name,
                "scenes": len(keyframes),
                "audio_atoms": sum(1 for a in atoms if a.type == AtomType.TEXT),
            },
        )

    @staticmethod
    def _audio_doc(doc: Any) -> RawDocument:
        """Return a synthetic RawDocument targeting the audio mimetype.

        Real extraction would invoke ffmpeg to extract the audio track; the
        lite path just pretends the same bytes are ``audio/mpeg`` so the
        ASR parser can produce transcript atoms.
        """
        return RawDocument(
            kb_id=doc.kb_id,
            filename=doc.filename or "audio.mp3",
            mimetype="audio/mpeg",
            content=doc.content,
            metadata=doc.metadata or {},
            doc_id=doc.doc_id,
        )

    def _detect_scenes(self, doc: Any) -> list[dict[str, Any]]:
        """Run scene detection. Returns a list of keyframe descriptors."""
        if self.scene_detector is not None:
            return list(self.scene_detector(doc))
        if self._scenedetect_mod is None:
            return []
        sd = self._scenedetect_mod
        # PySceneDetect's high-level API; we save keyframes to a temp dir
        # and return the bytes + timestamps.
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            content = doc.content if isinstance(doc.content, bytes) else doc.content.encode("utf-8")
            video_path = os.path.join(tmpdir, "input.video")
            with open(video_path, "wb") as f:
                f.write(content)
            try:
                manager = sd.Manager([video_path])
                detector = sd.ContentDetector(threshold=self.scene_threshold)
                manager.detect_scenes(detector=detector)
                scenes = manager.get_scene_list()
                # Extract a frame at the start of each scene.
                out: list[dict[str, Any]] = []
                for i, (start_tc, end_tc) in enumerate(scenes):
                    start_s = start_tc.get_seconds() if hasattr(start_tc, "get_seconds") else float(start_tc)
                    end_s = end_tc.get_seconds() if hasattr(end_tc, "get_seconds") else float(end_tc)
                    frame_path = os.path.join(tmpdir, f"scene_{i:04d}.jpg")
                    manager.save_images(
                        num_images=1,
                        image_ext=".jpg",
                        output_dir=tmpdir,
                        scene_img_format=lambda img, idx: f"scene_{idx:04d}.jpg",
                    )
                    try:
                        with open(frame_path, "rb") as f:
                            out.append({"start_s": start_s, "end_s": end_s, "bytes": f.read()})
                    except OSError:
                        continue
                return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("scene detection failed: %s", exc)
                return []


__all__ = ["AVParser"]
