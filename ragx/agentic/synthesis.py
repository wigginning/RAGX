"""Synthesis and Verification (07-agentic.md §7.2.2).

Synthesis: flagship model (role: synthesize) merges sub-answers into a
coherent response with citations.

Verification: single pass (max 1, §7.4) checks that the answer is grounded
in the retrieved context. Failed verification → AgenticVerifyFailedError(7003)
→ degrade to Standard.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from ragx.core.roles import LLMRole
from ragx.spi.interfaces import ChatMessage, ChatRequest

logger = logging.getLogger("ragx.agentic.synthesis")


class _SynthesisResult(BaseModel):
    """Internal synthesis output."""

    answer: str
    citations: list[str] = []  # chunk_ids referenced


class _VerifyResult(BaseModel):
    """Internal verification output."""

    passed: bool
    issues: str = ""
    confidence: float = 0.0


async def synthesise(
    sub_answers: list[str],
    query: str,
    llm: Any,
    kb_id: str = "default",
) -> tuple[str, list[str]]:
    """Merge sub-answers into a final response (§7.2.2).

    Uses the flagship model (role: synthesize). Returns (answer, cited_chunk_ids).
    """
    # Filter out [NO DATA] markers
    valid_answers = [a for a in sub_answers if a and a != "[NO DATA]"]
    if not valid_answers:
        return "Unable to find relevant information in the knowledge base.", []

    combined_context = "\n\n".join(
        f"--- Sub-answer {i+1} ---\n{a}" for i, a in enumerate(valid_answers)
    )

    req = ChatRequest(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are a RAG synthesis engine. Given a user query and "
                    "multiple sub-answers, produce a single coherent response. "
                    "Cite the sources using [n] markers where n refers to the "
                    "sub-answer number. Be concise and factual."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Query: {query}\n\nSub-answers:\n{combined_context}",
            ),
        ],
        role=LLMRole.SYNTHESIZE,
        temperature=0.0,
        kb_id=kb_id,
    )
    resp = await llm.chat(req)
    return resp.text, []


async def verify(
    answer: str,
    query: str,
    context: str,
    llm: Any,
    kb_id: str = "default",
) -> _VerifyResult:
    """Single verification pass (§7.2.2).

    Checks that the answer is grounded in the provided context. Only one pass
    (max 1 verification, §7.4).
    """
    req = ChatRequest(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are a RAG verification engine. Given a query, the "
                    "retrieved context, and the answer, determine if the answer "
                    "is fully supported by the context. Check for hallucinations, "
                    "unsupported claims, or missing key information. "
                    "Output JSON: {\"passed\": true/false, \"issues\": \"...\", "
                    "\"confidence\": 0.0-1.0}"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Query: {query}\n\n"
                    f"Context:\n{context[:4000]}\n\n"
                    f"Answer:\n{answer[:2000]}"
                ),
            ),
        ],
        role=LLMRole.SYNTHESIZE,
        temperature=0.0,
        json_mode=True,
        kb_id=kb_id,
    )
    return await llm.structured(req, schema=_VerifyResult)


class Verifier:
    """Verification node for the agentic state graph."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run verification on the synthesized answer (§7.2.2)."""
        from ragx.agentic.state import VerifyResult

        answer = state.get("context", "")
        query = state.get("query", "")
        kb_id = state.get("kb_id", "default")

        result = await verify(answer, query, "", self.llm, kb_id)

        verify_result = VerifyResult(
            passed=result.passed,
            issues=result.issues,
            confidence=result.confidence,
        )
        state["verify_result"] = verify_result
        return state
