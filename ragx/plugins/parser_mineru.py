"""MinerUParser (11-plugins-builtin.md §11.2.6, full profile).

Multimodal parser inspired by MinerU's "magic-pdf" approach. Where the
production MinerU invokes an end-to-end multimodal model, this implementation
focuses on the *structural* elements that benefit from a dedicated parser:

* Page-by-page reading-order extraction via PyMuPDF
* Detection of **table-like** regions (multi-row blocks with consistent column
  counts) and conversion to Markdown tables
* Detection of **formula-like** spans (lines that match a LaTeX-ish regex)
  and surfacing them as ``AtomType.FORMULA`` atoms with raw LaTeX
* All non-text content becomes ``AtomType.IMAGE`` placeholders; actual image
  bytes are stored by the caller via the ObjectStore (``payload_ref``)

Optional dependencies (declared in ``pyproject.toml [mineru]`` extra):

* ``pymupdf`` (fitz) — page iteration + drawing/text extraction
* ``pdfplumber`` — table detection (graceful degradation: no tables if missing)
"""

from __future__ import annotations

import logging
import re
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

logger = logging.getLogger("ragx.plugins.parser_mineru")

_NAME = "mineru"
_SUPPORTED_MIMETYPES: list[str] = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
]

# Heuristics for "this looks like a formula" — a single line that begins with
# one of these tokens. Real MinerU invokes a formula-recognition model; this
# regex is the lite fallback for the structural pass.
_FORMULA_TOKEN = re.compile(
    r"^\s*(\\\[|\\\(|\$|\\begin\{|\\frac|\\sum|\\int|\\sqrt|\\lim)",
)
_TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


def _maybe_import_pymupdf() -> Any:
    try:
        import fitz  # PyMuPDF

        return fitz
    except ImportError:
        return None


def _maybe_import_pdfplumber() -> Any:
    try:
        import pdfplumber  # type: ignore[import-not-found]

        return pdfplumber
    except ImportError:
        return None


def _maybe_import_pil() -> Any:
    try:
        from PIL import Image  # type: ignore[import-not-found]

        return Image
    except ImportError:
        return None


class MinerUParser:
    """Structural multimodal parser — page-aware reading order."""

    name: str = _NAME
    supported_mimetypes: list[str] = list(_SUPPORTED_MIMETYPES)

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.formula_enabled: bool = bool(self.config.get("formula_enabled", True))
        self.min_chars_per_page: int = int(self.config.get("min_chars_per_page", 8))

        self._fitz = _maybe_import_pymupdf()
        self._pdfplumber = _maybe_import_pdfplumber()
        self._pil = _maybe_import_pil()
        if self._fitz is None:
            raise PluginContractError(
                "MinerUParser requires pymupdf; install 'ragx[mineru]'",
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
                "mimetype not supported by the mineru parser",
                code=2002,
                details={
                    "mimetype": mt,
                    "supported": self.supported_mimetypes,
                    "parser": self.name,
                },
            )
        try:
            if mt == "application/pdf":
                return self._parse_pdf(doc)
            return self._parse_image(doc)
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise ParseError(
                "mineru parse failed",
                code=2001,
                details={"mimetype": mt, "error": str(exc)},
            ) from exc

    # -- PDFs --------------------------------------------------------------
    def _parse_pdf(self, doc: Any) -> ParseResult:
        content = doc.content if isinstance(doc.content, bytes) else doc.content.encode("utf-8")
        atoms: list[Atom] = []
        seq = 0
        with self._fitz.open(stream=content, filetype="pdf") as pdf:
            for page_index, page in enumerate(pdf, start=1):
                page_text = page.get_text("text") or ""
                if len(page_text.strip()) < self.min_chars_per_page:
                    # scanned-looking page — emit an IMAGE placeholder
                    atoms.append(self._image_atom(doc, seq, page_index))
                    seq += 1
                    continue
                # Table detection (pdfplumber, optional)
                table_atom = self._detect_table(doc, seq, page_index, content)
                if table_atom is not None:
                    atoms.append(table_atom)
                    seq += 1
                # Line-by-line: text vs formula
                for line in page_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if self.formula_enabled and _FORMULA_TOKEN.match(stripped):
                        atoms.append(self._formula_atom(doc, seq, page_index, stripped))
                    else:
                        atoms.append(self._text_atom(doc, seq, page_index, stripped))
                    seq += 1
        return ParseResult(
            atoms=atoms,
            metadata={"parser": self.name, "page_count": len(set(a.page for a in atoms if a.page))},
        )

    def _detect_table(
        self, doc: Any, seq: int, page_no: int, content: bytes
    ) -> Atom | None:
        """Heuristic: emit a TABLE atom if pdfplumber finds a table."""
        if self._pdfplumber is None:
            return None
        try:
            import io as _io

            with self._pdfplumber.open(_io.BytesIO(content)) as pdf:
                if page_no - 1 >= len(pdf.pages):
                    return None
                tables = pdf.pages[page_no - 1].find_tables()
                if not tables:
                    return None
                # Use the first table on the page.
                tbl = tables[0].extract()
                if not tbl or len(tbl) < 2:
                    return None
                md = self._table_to_markdown(tbl)
                if not md.strip():
                    return None
                return Atom(
                    atom_id=f"{doc.doc_id}#{seq:04d}",
                    doc_id=doc.doc_id or "",
                    type=AtomType.TABLE,
                    text=md,
                    content_hash=compute_atom_content_hash(md),
                    page=page_no,
                    context=None,
                    metadata={"parser": self.name},
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("table detection skipped on page %s: %s", page_no, exc)
            return None

    @staticmethod
    def _table_to_markdown(tbl: list[list[str | None]]) -> str:
        """Convert a pdfplumber table extract to a Markdown pipe-table."""
        rows: list[str] = []
        for i, row in enumerate(tbl):
            cells = [(c or "").replace("\n", " ").strip() for c in row]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("|" + "|".join("---" for _ in cells) + "|")
        return "\n".join(rows)

    # -- images ------------------------------------------------------------
    def _parse_image(self, doc: Any) -> ParseResult:
        # Single image: one IMAGE atom. A real MinerU run would call its
        # multimodal model for caption + structured elements; that step is
        # deferred to the VLM processor in the ingestion pipeline.
        return ParseResult(
            atoms=[self._image_atom(doc, 0, page_no=None)],
            metadata={"parser": self.name},
        )

    # -- helpers ------------------------------------------------------------
    def _text_atom(self, doc: Any, seq: int, page_no: int, text: str) -> Atom:
        return Atom(
            atom_id=f"{doc.doc_id}#{seq:04d}",
            doc_id=doc.doc_id or "",
            type=AtomType.TEXT,
            text=text,
            content_hash=compute_atom_content_hash(text),
            page=page_no,
            metadata={"parser": self.name},
        )

    def _formula_atom(self, doc: Any, seq: int, page_no: int, raw: str) -> Atom:
        # Strip the outer delimiters if present for the canonical text.
        body = raw.strip()
        if body.startswith("$$") and body.endswith("$$"):
            body = body[2:-2].strip()
        elif body.startswith("$") and body.endswith("$"):
            body = body[1:-1].strip()
        return Atom(
            atom_id=f"{doc.doc_id}#{seq:04d}",
            doc_id=doc.doc_id or "",
            type=AtomType.FORMULA,
            text=body,
            content_hash=compute_atom_content_hash(body),
            page=page_no,
            metadata={"parser": self.name},
        )

    def _image_atom(self, doc: Any, seq: int, page_no: int | None) -> Atom:
        content = doc.content if isinstance(doc.content, bytes) else doc.content.encode("utf-8")
        return Atom(
            atom_id=f"{doc.doc_id}#{seq:04d}",
            doc_id=doc.doc_id or "",
            type=AtomType.IMAGE,
            text=f"[image page={page_no}]",
            content_hash=compute_atom_content_hash(content),
            page=page_no,
            metadata={"parser": self.name, "payload_ref": doc.source_uri or None},
        )


__all__ = ["MinerUParser"]
