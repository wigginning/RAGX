"""FilterExpr validation (06-retrieval.md §6.6.1)."""

from __future__ import annotations

from ragx.core.exceptions import FilterValidationError
from ragx.core.models import FilterExpr

_VALID_OPS = {"eq", "ne", "gt", "ge", "lt", "le", "in", "contains"}


def validate_filter(expr: FilterExpr | None) -> None:
    """Validate a FilterExpr before pushdown. Raises ``FilterValidationError(4003)``."""
    if expr is None:
        return
    for clause in expr.as_list():
        op = str(clause.get("op"))
        field = str(clause.get("field"))
        if op not in _VALID_OPS:
            raise FilterValidationError(
                code=4003,
                message="invalid filter operator",
                details={"field": field, "op": op, "valid": sorted(_VALID_OPS)},
            )
        if not field:
            raise FilterValidationError(
                code=4003, message="filter field must be non-empty", details={"op": op}
            )
        if op == "in" and not isinstance(clause.get("value"), list):
            raise FilterValidationError(
                code=4003,
                message="'in' filter requires a list value",
                details={"field": field},
            )
