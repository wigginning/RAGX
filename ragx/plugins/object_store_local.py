"""LocalFSObjectStore (11-plugins-builtin.md §11.1.4).

Keeps binary atom payloads (images, formula rasters) on the local filesystem:

    key = f"{kb_id}/{doc_id}/{atom_id}.bin"

Not part of the SPI seven; it is an ingestion dependency injected into plugins
that produce ``payload_ref`` atoms (13-parsing.md).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from ragx.core.exceptions import StoreUnavailableError

_NAME = "local"


def key_for(kb_id: str, doc_id: str, atom_id: str) -> str:
    """Stable object key shared by all ObjectStore implementations."""
    return f"{kb_id}/{doc_id}/{atom_id}.bin"


class LocalFSObjectStore:
    name: str = _NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        root = cfg.get("root") or cfg.get("path") or "./data/objects"
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        safe = re.sub(r"[^0-9A-Za-z_./#-]", "_", key)
        path = (self.root / safe).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise StoreUnavailableError(
                "object key escapes the object store root",
                code=9001, details={"key": key},
            )
        return path

    async def put(self, key: str, data: bytes) -> str:
        path = self._path(key)

        def _write() -> str:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return key

        try:
            return await asyncio.to_thread(_write)
        except OSError as exc:
            raise StoreUnavailableError(
                "object write failed", code=9001, details={"key": key, "error": str(exc)}
            ) from exc

    async def get(self, key: str) -> bytes | None:
        path = self._path(key)

        def _read() -> bytes | None:
            return path.read_bytes() if path.exists() else None

        try:
            return await asyncio.to_thread(_read)
        except OSError as exc:
            raise StoreUnavailableError(
                "object read failed", code=9001, details={"key": key, "error": str(exc)}
            ) from exc

    async def delete(self, key: str) -> None:
        path = self._path(key)

        def _delete() -> None:
            if path.exists():
                path.unlink()

        await asyncio.to_thread(_delete)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def list_prefix(self, prefix: str) -> list[str]:
        base = (self.root / re.sub(r"[^0-9A-Za-z_./#-]", "_", prefix)).resolve()
        if not base.exists():
            return []
        root_resolved = self.root.resolve()
        return sorted(
            p.relative_to(root_resolved).as_posix()
            for p in base.rglob("*") if p.is_file()
        )

    async def startup(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def shutdown(self) -> None:
        return None
