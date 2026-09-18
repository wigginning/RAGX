"""Synthesis and Verification tests (07-agentic.md §7.2.2, §7.4).

Covers: ``synthesise`` merging sub-answers, ``[NO DATA]`` filtering, the
verification pass, and the Verifier node.
"""

from __future__ import annotations

from types import SimpleNamespace

from ragx.agentic.synthesis import Verifier, synthesise, verify


class _LLM:
    """Scripted chat / structured responses."""

    def __init__(self, *, chat_text: str = "综合答案", verify_passed: bool = True):
        self.chat_text = chat_text
        self.verify_passed = verify_passed
        self.chat_calls = 0
        self.structured_calls = 0

    async def chat(self, req):
        self.chat_calls += 1
        return SimpleNamespace(text=self.chat_text, usage=None)

    async def structured(self, req, *, schema):
        self.structured_calls += 1
        return SimpleNamespace(
            passed=self.verify_passed, issues="" if self.verify_passed else "unsupported",
            confidence=0.9,
        )


class TestSynthesise:
    async def test_merges_valid_sub_answers(self) -> None:
        llm = _LLM(chat_text="最终答案")
        answer, _ = await synthesise(
            ["子答案一", "子答案二"], "问题", llm, kb_id="kb_1"
        )
        assert answer == "最终答案"
        assert llm.chat_calls == 1

    async def test_filters_no_data_markers(self) -> None:
        """[NO DATA] markers are excluded from the synthesis context."""
        llm = _LLM()
        answer, _ = await synthesise(
            ["[NO DATA]", "真实子答案"], "问题", llm
        )
        assert answer == "综合答案"
        assert llm.chat_calls == 1

    async def test_all_no_data_returns_honest_answer_without_llm(self) -> None:
        llm = _LLM()
        answer, _ = await synthesise(["[NO DATA]", ""], "问题", llm)
        assert "Unable to find" in answer
        assert llm.chat_calls == 0


class TestVerify:
    async def test_passed(self) -> None:
        llm = _LLM(verify_passed=True)
        result = await verify("答案", "问题", "上下文", llm, kb_id="kb_1")
        assert result.passed is True
        assert result.confidence == 0.9

    async def test_failed_reports_issues(self) -> None:
        llm = _LLM(verify_passed=False)
        result = await verify("答案", "问题", "上下文", llm)
        assert result.passed is False
        assert result.issues == "unsupported"


class TestVerifierNode:
    async def test_run_sets_verify_result(self) -> None:
        from ragx.agentic.state import VerifyResult

        verifier = Verifier(_LLM(verify_passed=True))
        state = {"query": "问题", "context": "答案", "kb_id": "kb_1"}
        out = await verifier.run(state)

        assert isinstance(out["verify_result"], VerifyResult)
        assert out["verify_result"].passed is True

    async def test_run_marks_failed(self) -> None:
        verifier = Verifier(_LLM(verify_passed=False))
        state = {"query": "问题", "context": "答案", "kb_id": "kb_1"}
        out = await verifier.run(state)
        assert out["verify_result"].passed is False
