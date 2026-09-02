"""Re-export of the canonical :class:`LLMRole`.

The canonical definition lives in ``ragx.core.roles`` (see its docstring for
why); this module keeps the documented import path ``ragx.llm.roles.LLMRole``
working (08-llm.md §8.2.1).
"""

from ragx.core.roles import LLMRole

__all__ = ["LLMRole"]
