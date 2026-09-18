"""OpenAICompatEmbedder - embeddings from any OpenAI-compatible endpoint.

Same transport assumptions as :class:`OpenAICompatibleLLM`
(11-plugins-builtin.md §11.1.5): ``/v1/embeddings``, ``base_url`` + API key
read from the environment variable named in the config (the key itself is never
stored in configuration).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ragx.core.exceptions import LLMError, PluginContractError

_NAME = "openai_compat"
DEFAULT_TIMEOUT_S = 30.0


def _resolve_api_key(config: dict[str, Any]) -> str:
    key_env = config.get("api_key_env") or config.get("api_key")
    if not key_env:
        return ""
    value = os.environ.get(str(key_env), "")
    if value:
        return value
    # tolerate a literal key in local/dev configs
    return str(key_env)


def _base_url(config: dict[str, Any]) -> str:
    return str(config.get("base_url", "https://api.openai.com/v1")).rstrip("/")


class OpenAICompatEmbedder:
    """SPI ``Embedder`` for OpenAI-compatible ``/v1/embeddings`` services."""

    name: str = _NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._base_url = _base_url(cfg)
        self._api_key = _resolve_api_key(cfg)
        self._model: str = str(cfg.get("model", "text-embedding-3-small"))
        self._timeout_s: float = float(cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.dimension: int = int(cfg.get("dim", 1536))
        self.max_batch_size: int = int(cfg.get("max_batch_size", 64))
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url, timeout=self._timeout_s, headers=headers
            )
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._http()
        try:
            out: list[list[float]] = []
            for start in range(0, len(texts), self.max_batch_size):
                batch = texts[start : start + self.max_batch_size]
                payload = {"model": self._model, "input": batch}
                try:
                    resp = await client.post("/embeddings", json=payload)
                except httpx.TimeoutException as exc:
                    raise LLMError(
                        "embedder endpoint timed out", details={"error": str(exc)}
                    ) from exc
                except httpx.HTTPError as exc:
                    raise LLMError(
                        "embedder endpoint unreachable", details={"error": str(exc)}
                    ) from exc
                if resp.status_code != 200:
                    raise LLMError(
                        "embedder endpoint returned an error",
                        details={"status": resp.status_code, "body": resp.text[:500]},
                    )
                data = resp.json().get("data", [])
                if len(data) != len(batch):
                    raise PluginContractError(
                        "embedding response length mismatch",
                        details={"expected": len(batch), "got": len(data)},
                    )
                for item in sorted(data, key=lambda d: d.get("index", 0)):
                    vec = list(item.get("embedding") or [])
                    if len(vec) != self.dimension:
                        raise PluginContractError(
                            "embedding dimension mismatch",
                            details={"expected": self.dimension, "got": len(vec),
                                     "model": self._model},
                        )
                    out.append([float(x) for x in vec])
            return out
        except (LLMError, PluginContractError):
            raise
        except Exception as exc:
            raise LLMError("embedding failed", details={"error": str(exc)}) from exc

    async def startup(self) -> None:
        self._http()

    async def shutdown(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
