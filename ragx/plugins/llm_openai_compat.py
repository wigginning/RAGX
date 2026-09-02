"""OpenAICompatibleLLM (11-plugins-builtin.md §11.1.5).

Talks to any OpenAI-compatible chat-completions endpoint (vLLM, Ollama, OpenAI,
NIM). Third-party failures are translated at the plugin boundary into the RAGX
exception tree (02-core.md §2.3); ``details["retryable"]`` tells the Resilient
Router whether the failure is worth a retry (08-llm.md §8.3).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from ragx.core.exceptions import (
    LLMError,
    PluginContractError,
    RateLimitError,
    StructuredParseError,
)
from ragx.core.models import TokenUsage
from ragx.spi.interfaces import (
    ChatChunk,
    ChatRequest,
    ChatResponse,
    LLMCapabilities,
)

_NAME = "openai_compat"
DEFAULT_TIMEOUT_S = 30.0
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def _resolve_api_key(config: dict[str, Any]) -> str:
    """The key itself is never stored in config - only the env var name."""
    key_env = config.get("api_key_env") or config.get("api_key")
    if not key_env:
        return ""
    value = os.environ.get(str(key_env), "")
    return value if value else str(key_env)


def _base_url(config: dict[str, Any]) -> str:
    return str(config.get("base_url", "https://api.openai.com/v1")).rstrip("/")


def _extract_usage(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion, total=total)


def _translate_http_error(resp: httpx.Response) -> Exception:
    """Map an HTTP failure to a RAGX domain error with a retryable flag."""
    status = resp.status_code
    body = resp.text[:500]
    if status == 429:
        return RateLimitError(
            "provider rate limited", details={"status": 429, "body": body,
                                              "retryable": True}
        )
    if 500 <= status < 600:
        return LLMError(
            "provider returned a server error",
            details={"status": status, "body": body, "retryable": True},
        )
    return LLMError(
        "provider rejected the request",
        details={"status": status, "body": body, "retryable": False},
    )


class OpenAICompatibleLLM:
    """SPI ``LLMProvider`` for OpenAI-compatible endpoints."""

    name: str = _NAME
    capabilities: LLMCapabilities = LLMCapabilities(
        supports_json_mode=True, supports_stream=True
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._base_url = _base_url(cfg)
        self._api_key = _resolve_api_key(cfg)
        self._default_model: str = str(cfg.get("model", "gpt-4o-mini"))
        self._timeout_s: float = float(cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
        self._max_retries_prompt_fallback = int(cfg.get("structured_retries", 1))
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

    def _payload(self, req: ChatRequest, *, json_mode: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model or self._default_model,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature,
            "stream": False,
        }
        if req.max_tokens:
            payload["max_tokens"] = req.max_tokens
        if json_mode and self.capabilities.supports_json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    @staticmethod
    def _parse_schema(text: str, schema: type[BaseModel]) -> BaseModel:
        """Extract + validate a JSON object from an LLM response."""
        candidate = text.strip()
        block = _JSON_BLOCK.search(candidate)
        if block:
            candidate = block.group(0)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise StructuredParseError(
                "model output is not valid JSON", details={"error": str(exc),
                                                           "head": text[:200]}
            ) from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise StructuredParseError(
                "model output violates the schema", details={"errors": exc.errors()[:5]}
            ) from exc

    # -- SPI ---------------------------------------------------------------
    async def chat(self, req: ChatRequest) -> ChatResponse:
        client = self._http()
        try:
            resp = await client.post("/chat/completions", json=self._payload(req))
        except httpx.TimeoutException as exc:
            raise LLMError(
                "provider timed out", details={"error": str(exc), "retryable": True}
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "provider unreachable", details={"error": str(exc), "retryable": True}
            ) from exc
        if resp.status_code != 200:
            raise _translate_http_error(resp)
        try:
            payload = resp.json()
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise PluginContractError(
                "unexpected provider response shape", details={"error": str(exc)}
            ) from exc
        return ChatResponse(
            text=text,
            usage=_extract_usage(payload),
            model=payload.get("model"),
            raw=payload,
        )

    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        if not self.capabilities.supports_stream:
            return
        client = self._http()
        payload = self._payload(req)
        payload["stream"] = True
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _translate_http_error(resp)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        yield ChatChunk(delta="", finish_reason="stop")
                        return
                    try:
                        piece = json.loads(data)
                        choice = piece["choices"][0]
                    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                        raise PluginContractError(
                            "malformed SSE frame", details={"error": str(exc)}
                        ) from exc
                    delta = choice.get("delta", {}).get("content") or ""
                    if delta:
                        yield ChatChunk(delta=delta)
        except httpx.TimeoutException as exc:
            raise LLMError(
                "provider timed out", details={"error": str(exc), "retryable": True}
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "provider unreachable", details={"error": str(exc), "retryable": True}
            ) from exc

    async def structured(
        self, req: ChatRequest, *, schema: type[BaseModel]
    ) -> BaseModel:
        """Structured output with json_mode when available, else prompt + retry."""
        use_json_mode = self.capabilities.supports_json_mode and req.json_mode
        hint = schema.model_json_schema()
        prompted = req.model_copy(deep=True)
        hinted_messages = list(prompted.messages) + [
            prompted.messages[-1].model_copy(update={
                "content": prompted.messages[-1].content
                + f"\n\n只输出符合以下 JSON Schema 的 JSON 对象，不要输出任何其它文字：{json.dumps(hint, ensure_ascii=False)}"
            })
        ]
        prompted = prompted.model_copy(update={"messages": hinted_messages})
        last_error: Exception | None = None
        for _ in range(1 + self._max_retries_prompt_fallback):
            response = await self.chat(
                prompted.model_copy(update={"json_mode": use_json_mode})
            )
            try:
                return self._parse_schema(response.text, schema)
            except StructuredParseError as exc:
                last_error = exc
                continue
        assert last_error is not None
        raise last_error

    async def startup(self) -> None:
        self._http()

    async def shutdown(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
