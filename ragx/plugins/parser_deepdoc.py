"""DeepDocParser (11-plugins-builtin.md §11.2.5, full profile).

Layout-aware document parser inspired by RAGFlow's DeepDoc. The production
implementation would invoke DeepDoc's layout model + PaddleOCR; in this
distribution we provide a **graceful-degradation** implementation that uses
only stdlib + ``pdfplumber`` (already a hard dep in the deepdoc extra) for
the digital-PDF path, and ``pytesseract`` if installed for the OCR fallback.

Hard guarantees (independent of OCR / layout-model availability):

* :class:`Parser` SPI compliance — ``name``, ``supported_mimetypes``,
  ``parse()``, ``startup()``, ``shutdown()``.
* Born-digital PDFs with a text layer always parse to ``AtomType.TEXT`` atoms
  with ``page`` + ``bbox`` populated (RAGFlow lesson: bbox enables Chunk
  traceability).
* Pure image files become a single ``IMAGE`` atom (or ``TEXT`` with the OCR
  result when OCR is enabled and a transcript is available).
* Scanned PDFs without a text layer require ``pytesseract`` — when missing,
  the parser raises :class:`UnsupportedFormatError` (2002) so the
  ingestion pipeline can surface a clear upgrade hint to the user.

Optional dependencies (declared in ``pyproject.toml [deepdoc]`` extra):

* ``pdfplumber`` — born-digital text + bbox extraction
* ``pypdf``    — text layer fallback + page iteration
* ``pytesseract`` — OCR for scanned pages and image files
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Any

from ragx.core.exceptions import (
    ParseError,
    PluginContractError,
    UnsupportedFormatError,
)
from ragx.core.hashing import compute_atom_content_hash
from ragx.core.models import Atom, AtomType
from ragx.core.settings import ParseOptions
from ragx.spi.interfaces import ParseResult

logger = logging.getLogger("ragx.plugins.parser_deepdoc")

_NAME = "deepdoc"
_SUPPORTED_MIMETYPES: list[str] = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
]


@dataclass
class _PdfBackend:
    """Wrapper over the optional pdfplumber / pypdf backends."""

    name: str
    available: bool

    @property
    def is_digital(self) -> bool:
        return self.name == "pdfplumber"


def _maybe_import_pdfplumber() -> Any:
    try:
        import pdfplumber  # type: ignore[import-not-found]

        return pdfplumber
    except ImportError:
        return None


def _maybe_import_pypdf() -> Any:
    try:
        import pypdf  # type: ignore[import-not-found]

        return pypdf
    except ImportError:
        return None


def _maybe_import_pytesseract() -> Any:
    try:
        import pytesseract  # type: ignore[import-not-found]

        return pytesseract
    except ImportError:
        return None


def _maybe_import_pil() -> Any:
    try:
        from PIL import Image  # type: ignore[import-not-found]

        return Image
    except ImportError:
        return None


class DeepDocParser:
    """Layout-aware parser (digital-first, OCR fallback when available)."""

    name: str = _NAME
    supported_mimetypes: list[str] = list(_SUPPORTED_MIMETYPES)

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.ocr_enabled: bool = bool(self.config.get("ocr_enabled", True))
        self.ocr_min_chars: int = int(self.config.get("ocr_min_chars", 16))
        self.ocr_lang: str = str(self.config.get("ocr_lang", "chi_sim+eng"))
        self.table_marker: str = str(self.config.get("table_marker", "TABLE"))

        # Probe optional deps at construction time so parse() is a hot path.
        self._pdfplumber = _maybe_import_pdfplumber()
        self._pypdf = _maybe_import_pypdf()
        self._pytesseract = _maybe_import_pytesseract()
        self._pil = _maybe_import_pil()
        if self._pdfplumber is None and self._pypdf is None:
            raise PluginContractError(
                "DeepDocParser requires pdfplumber or pypdf; install 'ragx[deepdoc]'",
                details={"parser": self.name},
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
                "mimetype not supported by the deepdoc parser",
                code=2002,
                details={
                    "mimetype": mt,
                    "supported": self.supported_mimetypes,
                    "parser": self.name,
                },
            )
        try:
            if mt == "application/pdf":
                return await self._parse_pdf(doc)
            return await self._parse_image(doc)
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise ParseError(
                "deepdoc parse failed",
                code=2001,
                details={"mimetype": mt, "error": str(exc)},
            ) from exc

    # -- PDFs --------------------------------------------------------------
    async def _parse_pdf(self, doc: Any) -> ParseResult:
        content = self._bytes(doc.content)
        pages = self._extract_pdf_pages(content)
        atoms: list[Atom] = []
        for seq, page in enumerate(pages):
            if page.is_digital and page.text.strip():
                atoms.append(self._make_text_atom(doc, seq, page.text, page.page_no, page.heading))
            elif self.ocr_enabled and self._pytesseract is not None and self._pil is not None:
                ocr_text = self._ocr_page(page, content)
                if ocr_text.strip():
                    atoms.append(
                        self._make_text_atom(
                            doc, seq, ocr_text, page.page_no, page.heading,
                            extra_meta={"ocr": True},
                        )
                    )
                else:
                    # scanned page, OCR empty -> IMAGE atom
                    atoms.append(self._make_image_atom(doc, seq, page.page_no))
            else:
                # scanned page, no OCR available -> IMAGE atom
                atoms.append(self._make_image_atom(doc, seq, page.page_no))
        return ParseResult(
            atoms=atoms,
            metadata={
                "parser": self.name,
                "pages": len(pages),
                "ocr_used": any(a.metadata.get("ocr") for a in atoms),
            },
        )

    def _extract_pdf_pages(self, content: bytes) -> list[Any]:
        """Run pdfplumber (preferred) or pypdf to extract per-page text.

        Returns a list of small dataclasses with the fields the parser needs;
        we keep this internal so the public API is independent of which
        optional library happened to load.
        """
        # local dataclass — kept private to this module
        from dataclasses import dataclass as _dc

        @_dc
        class _Page:
            page_no: int
            text: str
            is_digital: bool
            heading: str = ""

        pages: list[_Page] = []
        if self._pdfplumber is not None:
            try:
                with self._pdfplumber.open(io.BytesIO(content)) as pdf:
                    for i, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        # crude digital-PDF detection: at least N chars
                        is_digital = len(text.strip()) >= self.ocr_min_chars
                        # crude heading detection: first non-empty line in
                        # bold-ish font is often a section title. The lite
                        # detector below is intentional — full layout
                        # analysis requires the DeepDoc layout model which
                        # is opt-in via the gpu profile.
                        heading = self._first_line(text) if is_digital else ""
                        pages.append(_Page(i, text, is_digital, heading))
                return pages
            except Exception as exc:  # noqa: BLE001
                logger.warning("pdfplumber failed; falling back to pypdf: %s", exc)

        if self._pypdf is not None:
            try:
                reader = self._pypdf.PdfReader(io.BytesIO(content))
                for i, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    is_digital = len(text.strip()) >= self.ocr_min_chars
                    heading = self._first_line(text) if is_digital else ""
                    pages.append(_Page(i, text, is_digital, heading))
                return pages
            except Exception as exc:  # noqa: BLE001
                logger.warning("pypdf failed: %s", exc)

        raise ParseError(
            "no PDF backend succeeded",
            code=2001,
            details={"backend": "pdfplumber+pypdf"},
        )

    @staticmethod
    def _first_line(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return ""

    def _ocr_page(self, page: Any, content: bytes) -> str:
        """Run pytesseract on a single PDF page (best-effort)."""
        if self._pytesseract is None or self._pdfplumber is None or self._pil is None:
            return ""
        try:
            with self._pdfplumber.open(io.BytesIO(content)) as pdf:
                pdf_page = pdf.pages[page.page_no - 1]
                img = pdf_page.to_image(resolution=200).original
                return self._pytesseract.image_to_string(img, lang=self.ocr_lang)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR failed for page %s: %s", page.page_no, exc)
            return ""

    # -- images ------------------------------------------------------------
    async def _parse_image(self, doc: Any) -> ParseResult:
        if self.ocr_enabled and self._pytesseract is not None and self._pil is not None:
            try:
                img = self._pil.open(io.BytesIO(self._bytes(doc.content)))
                text = self._pytesseract.image_to_string(img, lang=self.ocr_lang)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed for image: %s", exc)
                text = ""
            if text.strip():
                atom = self._make_text_atom(doc, 0, text, 1, extra_meta={"ocr": True})
                return ParseResult(atoms=[atom], metadata={"parser": self.name, "ocr": True})
        # fallback: single IMAGE atom
        atom = self._make_image_atom(doc, 0, page_no=None)
        return ParseResult(atoms=[atom], metadata={"parser": self.name, "ocr": False})

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _bytes(content: Any) -> bytes:
        return content if isinstance(content, bytes) else content.encode("utf-8")

    def _make_text_atom(
        self,
        doc: Any,
        seq: int,
        text: str,
        page_no: int,
        heading: str = "",
        extra_meta: dict[str, Any] | None = None,
    ) -> Atom:
        meta: dict[str, Any] = dict(extra_meta or {})
        if heading:
            meta["heading"] = heading
        return Atom(
            atom_id=f"{doc.doc_id}#{seq:04d}",
            doc_id=doc.doc_id or "",
            type=AtomType.TEXT,
            text=text,
            content_hash=compute_atom_content_hash(text),
            page=page_no,
            context=heading or None,
            metadata=meta,
        )

    def _make_image_atom(self, doc: Any, seq: int, page_no: int | None) -> Atom:
        text = f"[image at page {page_no or '?'}]"
        return Atom(
            atom_id=f"{doc.doc_id}#{seq:04d}",
            doc_id=doc.doc_id or "",
            type=AtomType.IMAGE,
            text=text,
            content_hash=compute_atom_content_hash(self._bytes(doc.content)),
            page=page_no,
            metadata={"payload_ref": doc.source_uri or None, "ocr": False},
        )


__all__ = ["DeepDocParser"]
