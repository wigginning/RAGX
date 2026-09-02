"""TextMarkdownParser (11-plugins-builtin.md §11.1.1, lite profile).

Pure text / Markdown parsing with no third-party dependency:

* splits on the configured heading levels into sections; ``Atom.context`` holds
  the heading path so chunks stay understandable out of context
  (04-chunking.md §4.3.5 ``context_prefix``)
* paragraphs are grouped into atoms of roughly ``target_tokens`` so the chunker
  mostly merges instead of hard-splitting
* ``atom_id == f"{doc_id}#{seq:04d}"`` in reading order (13-parsing.md)
* Markdown tables are kept as markdown text (single ``Atom.type = TEXT``)
"""

from __future__ import annotations

import re

from ragx.core.exceptions import ParseError, UnsupportedFormatError
from ragx.core.hashing import compute_atom_content_hash
from ragx.core.models import Atom, AtomType
from ragx.core.settings import ParseOptions
from ragx.core.tokens import count_tokens
from ragx.spi.interfaces import ParseResult

_NAME = "text"
_MIMETYPES: list[str] = ["text/plain", "text/markdown", "text/x-markdown"]

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")


def _decode(content: bytes | str) -> str:
    if isinstance(content, str):
        return content
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError(
        "unable to decode document bytes", code=2001,
        details={"tried": ["utf-8", "utf-8-sig", "gb18030", "latin-1"]},
    )


def _heading_levels(config: dict) -> list[int]:
    raw = (config or {}).get("heading_levels")
    if isinstance(raw, list) and raw:
        return sorted({int(x) for x in raw if 1 <= int(x) <= 6})
    return [1, 2, 3]


def _is_table_block(lines: list[str]) -> bool:
    return len(lines) >= 2 and _TABLE_ROW.match(lines[0]) and _TABLE_SEP.match(lines[1])


class TextMarkdownParser:
    """SPI ``Parser`` for plain text and Markdown."""

    name: str = _NAME
    supported_mimetypes: list[str] = list(_MIMETYPES)

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._levels = _heading_levels(self.config)
        self._target_tokens = int(self.config.get("target_tokens", 512))
        self._max_tokens = int(self.config.get("max_tokens", 768))
        self.markdown = bool(self.config.get("markdown", True))

    # -- SPI ---------------------------------------------------------------
    async def parse(self, doc, *, options: ParseOptions) -> ParseResult:  # noqa: ANN001
        if doc.mimetype not in self.supported_mimetypes:
            raise UnsupportedFormatError(
                "mimetype not supported by the text parser",
                code=2002,
                details={"mimetype": doc.mimetype,
                         "supported": self.supported_mimetypes,
                         "parser": self.name},
            )
        try:
            text = _decode(doc.content)
            if not text.strip():
                return ParseResult(atoms=[])
            sections = self._split_sections(text, markdown=self.markdown)
            groups = self._group_to_atoms(sections)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(
                "text parsing failed", code=2001, details={"error": str(exc)}
            ) from exc

        atoms: list[Atom] = []
        for seq, (block, context) in enumerate(groups):
            atoms.append(
                Atom(
                    atom_id=f"{doc.doc_id}#{seq:04d}",
                    doc_id=doc.doc_id or "",
                    type=AtomType.TEXT,
                    text=block,
                    content_hash=compute_atom_content_hash(block),
                    page=1,
                    context=context or None,
                )
            )
        return ParseResult(atoms=atoms, metadata={"parser": self.name, "bytes": len(doc.content if isinstance(doc.content, bytes) else doc.content.encode("utf-8"))})

    async def startup(self) -> None:  # pragma: no cover - trivial hook
        return None

    async def shutdown(self) -> None:  # pragma: no cover - trivial hook
        return None

    # -- section splitting -------------------------------------------------
    def _split_sections(self, text: str, *, markdown: bool) -> list[tuple[str, str]]:
        """Return ``[(section_text, heading_path)]`` in reading order."""
        sections: list[tuple[str, str]] = []
        lines = text.splitlines()
        buffer: list[str] = []
        stack: list[str] = []          # active heading path
        in_fence = False
        table: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if table:
                buffer.extend(table)
                table.clear()
            joined = "\n".join(buffer).strip("\n")
            if joined.strip():
                sections.append((joined, " > ".join(stack)))
            buffer.clear()

        for line in lines:
            if _FENCE.match(line.strip()):
                in_fence = not in_fence
                table.clear()
                buffer.append(line)
                continue
            if markdown and not in_fence:
                m = _HEADING.match(line)
                if m:
                    level = len(m.group(1))
                    title = m.group(2).strip()
                    if level in self._levels:
                        flush()
                        stack = stack[: level - 1] + [title]
                    else:
                        buffer.append(line)
                    continue
            if _TABLE_ROW.match(line):
                if _TABLE_SEP.match(line) and table:
                    table.append(line)
                elif table and _TABLE_ROW.match(line):
                    table.append(line)
                else:
                    flush()
                    table = [line]
                continue
            if table:
                table.append(line)
                continue
            buffer.append(line)
        flush()
        if not sections:
            joined = text.strip()
            if joined:
                sections.append((joined, ""))
        return sections

    # -- atom grouping -----------------------------------------------------
    def _group_to_atoms(self, sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge paragraphs until an atom reaches ``target_tokens``."""
        out: list[tuple[str, str]] = []
        for block, context in sections:
            chunks: list[str] = []
            size = 0
            for paragraph in re.split(r"\n\s*\n", block):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                para_tokens = count_tokens(paragraph)
                # a table or code block is kept whole when it fits, otherwise split
                if size + para_tokens > self._target_tokens and chunks:
                    out.append(("\n\n".join(chunks), context))
                    chunks, size = [], 0
                if para_tokens > self._max_tokens and not chunks:
                    out.extend(self._hard_split(paragraph, context))
                    continue
                chunks.append(paragraph)
                size += para_tokens
            if chunks:
                out.append(("\n\n".join(chunks), context))
        return out

    @staticmethod
    def _hard_split(text: str, context: str) -> list[tuple[str, str]]:
        """Sentence-wise hard split for oversized paragraphs."""
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s for s in sentences if s.strip()]
        groups: list[tuple[str, str]] = []
        buf: list[str] = []
        size = 0
        for sentence in sentences:
            t = count_tokens(sentence)
            if size + t > 768 and buf:
                groups.append(("\n".join(buf), context))
                buf, size = [], 0
            buf.append(sentence)
            size += t
        if buf:
            groups.append(("\n".join(buf), context))
        return groups
