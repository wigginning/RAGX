"""POST /v1/chat/completions (09-api.md §9.4.1).

OpenAI-compatible request with a ``ragx`` extension object. Supports both SSE
streaming and non-streaming responses.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragx.api.middleware import require_kb_access
from ragx.core.models import RequestOverride

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


class RagxExt(BaseModel):
    kb_id: str
    mode: str = "auto"
    overrides: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool = True
    temperature: float = 0.2
    max_tokens: int | None = None
    ragx: RagxExt


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    require_kb_access(request, req.ragx.kb_id)
    service = request.app.state.query_service
    embedder = request.app.state.embedder
    trace_id = request.state.trace_id
    query = req.messages[-1].content if req.messages else ""
    override = RequestOverride.from_dict(req.ragx.overrides)
    qvec = (await embedder.embed([query]))[0]

    result = await service.query(
        query, qvec, kb_id=req.ragx.kb_id, trace_id=trace_id, override=override
    )

    if not req.stream:
        return {
            "id": f"chatcmpl-{trace_id}",
            "object": "chat.completion",
            "model": req.model or "ragx-default",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total,
            },
            "ragx": {
                "mode": result.mode,
                "citations": [c.model_dump() for c in result.citations],
                "cost_usd": result.cost_usd,
                "degraded": result.degraded,
                "trace_id": trace_id,
            },
        }

    async def _stream() -> AsyncIterator[str]:
        yield _sse("role", {"role": "assistant"})
        yield _sse("content", {"delta": result.answer})
        yield _sse(
            "ragx.citations",
            {"citations": [c.model_dump() for c in result.citations]},
        )
        yield _sse(
            "ragx.usage",
            {
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total": result.usage.total,
                },
                "cost_usd": result.cost_usd,
                "mode": result.mode,
                "degraded": result.degraded,
            },
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
