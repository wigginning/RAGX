"""Token usage collection (08-llm.md §8.7).

When a provider omits ``usage``, estimate with a token counter and mark the
result ``estimated=True`` so observability can distinguish real metering from
estimation.
"""

from __future__ import annotations

from ragx.core.models import TokenUsage
from ragx.core.tokens import count_tokens
from ragx.spi.interfaces import ChatResponse


def collect_usage(resp: ChatResponse, model: str, prompt_text: str) -> TokenUsage:
    """Return the provider-reported usage, or an estimate when it is missing."""
    u = resp.usage
    if u is None or u.total == 0:
        prompt = count_tokens(prompt_text)
        completion = count_tokens(resp.text)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total=prompt + completion,
            estimated=True,
        )
    return u
