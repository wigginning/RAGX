"""Auth middleware (09-api.md §9.2.1 / §9.2.2).

Primary auth is an API key carried in ``Authorization: Bearer <key>`` or
``X-API-Key: <key>``. The key itself is never stored — only ``sha256(key)`` is
matched (against ``Settings.security.api_keys`` static seed and/or the
Metadata DB ``api_keys`` table). An authenticated request exposes
``request.state.auth`` (an :class:`AuthContext`) and
``request.state.tenant_id``; route handlers enforce kb access with
:func:`require_kb_access` (403 when ``kb_id`` is outside the key's ``kb_acl``).

Optional JWT (``security.jwt_enabled``) is verified with HS256 using only the
standard library (``hmac`` + ``base64``), so the lite profile gains JWT support
without a new dependency. JWT and API key are mutually exclusive — a credential
is treated as a JWT only when it has three dot-separated segments.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ragx.api.errors import error_response
from ragx.core.exceptions import AuthError, RAGXError
from ragx.core.settings import SecurityConfig

#: kb_acl semantics: an empty/missing ACL is treated as "unrestricted" for v1
#: convenience (a key created without an explicit allow-list can reach any kb);
#: a non-empty list is an allow-list (09-api.md §9.2.1).
_UNRESTRICTED: tuple = (None, [], ())


@dataclass
class AuthContext:
    """Resolved identity attached to ``request.state.auth``."""

    authenticated: bool
    key_id: str = "anonymous"
    tenant_id: str = "default"
    kb_acl: list[str] | None = None  # None => unrestricted


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_credential(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def _looks_like_jwt(token: str) -> bool:
    return token.count(".") == 2


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def verify_jwt(token: str, secret: str, *, now: float | None = None) -> dict[str, Any]:
    """Verify an HS256 JWT and return its claims (stdlib only).

    Raises ``AuthError(1002)`` on any malformation, signature mismatch or
    expiry (09-api.md §9.2.2).
    """
    try:
        header_seg, payload_seg, sig_seg = token.split(".")
        header = json.loads(_b64url_decode(header_seg))
        payload = json.loads(_b64url_decode(payload_seg))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthError("invalid JWT", code=1002, details={"error": str(exc)}) from exc

    if header.get("alg") != "HS256":
        raise AuthError("unsupported JWT algorithm", code=1002,
                        details={"alg": header.get("alg")})
    signing_input = f"{header_seg}.{payload_seg}".encode()
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        provided = _b64url_decode(sig_seg)
    except Exception as exc:  # noqa: BLE001 - malformed signature segment
        raise AuthError("invalid JWT signature", code=1002,
                        details={"error": str(exc)}) from exc
    if not hmac.compare_digest(expected, provided):
        raise AuthError("invalid JWT signature", code=1002)

    ts = now if now is not None else time.time()
    if "exp" in payload and float(payload["exp"]) < ts:
        raise AuthError("JWT expired", code=1002)
    return payload


class AuthMiddleware(BaseHTTPMiddleware):
    """Resolve the request identity from API key or JWT (§9.2.1 / §9.2.2)."""

    def __init__(
        self,
        app: Any,
        *,
        security: SecurityConfig,
        store: Any = None,  # MetadataStore (optional; api_keys table lookup)
    ) -> None:
        super().__init__(app)
        self.security = security
        self.store = store

    @property
    def enabled(self) -> bool:
        """Auth is enforced only when keys are configured or JWT is on.

        A bare lite profile (no keys, no JWT) stays open so the critical path
        and lite compose smoke run without credentials.
        """
        return bool(self.security.api_keys) or self.security.jwt_enabled

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            request.state.auth = await self._resolve(request)
            request.state.tenant_id = request.state.auth.tenant_id
        except RAGXError as exc:
            # Middleware sits outside FastAPI's ExceptionMiddleware, so convert
            # to the unified envelope here (§9.1.2) instead of re-raising.
            return error_response(exc, getattr(request.state, "trace_id", None))
        return await call_next(request)

    async def _resolve(self, request: Request) -> AuthContext:
        if not self.enabled:
            return AuthContext(authenticated=False)

        credential = _extract_credential(request)
        if not credential:
            raise AuthError(
                "missing credentials",
                code=1002,
                details={"hint": "provide Authorization: Bearer <key> or X-API-Key"},
            )

        # JWT takes precedence when the token is JWT-shaped and JWT is enabled
        # (09-api.md §9.2.2: JWT and API key are mutually exclusive).
        if self.security.jwt_enabled and _looks_like_jwt(credential):
            claims = verify_jwt(credential, self.security.jwt_secret)
            return AuthContext(
                authenticated=True,
                key_id=f"jwt:{claims.get('sub', '')}",
                tenant_id=str(claims.get("tenant", "default")),
                kb_acl=claims.get("kb_acl"),
            )

        ctx = await self._resolve_api_key(credential)
        if ctx is None:
            raise AuthError(
                "invalid API key", code=1002,
                details={"key_prefix": credential[:8]},
            )
        return ctx

    async def _resolve_api_key(self, credential: str) -> AuthContext | None:
        key_hash = _sha256(credential)

        # 1) static seed from settings
        for key_id, spec in self.security.api_keys.items():
            stored_hash = spec.get("key_hash")
            if stored_hash is None and "key" in spec:
                stored_hash = _sha256(str(spec["key"]))
            # Constant-time: don't let response timing leak the stored digest.
            if stored_hash is not None and hmac.compare_digest(stored_hash, key_hash):
                if not spec.get("enabled", True):
                    raise AuthError("API key disabled", code=1002,
                                    details={"key_id": key_id})
                return AuthContext(
                    authenticated=True,
                    key_id=key_id,
                    tenant_id=str(spec.get("tenant_id", "default")),
                    kb_acl=list(spec.get("kb_acl") or []) or None,
                )

        # 2) metadata DB api_keys table
        if self.store is not None:
            row = await self.store.get_api_key_by_hash(key_hash)
            if row is not None:
                if not row["enabled"]:
                    raise AuthError("API key disabled", code=1002,
                                    details={"key_id": row["key_id"]})
                return AuthContext(
                    authenticated=True,
                    key_id=row["key_id"],
                    tenant_id=row["tenant_id"],
                    kb_acl=row["kb_acl"] or None,
                )
        return None


def require_kb_access(request: Request, kb_id: str) -> None:
    """Raise ``AuthError(1003)`` when ``kb_id`` is outside the key's ACL.

    Route handlers that operate on a knowledge base call this after parsing the
    request. When auth is disabled (anonymous context) access is unrestricted.
    """
    auth: AuthContext | None = getattr(request.state, "auth", None)
    if auth is None or not auth.authenticated:
        return
    acl = auth.kb_acl
    if not acl:  # None / empty allow-list => unrestricted (v1 semantics)
        return
    if kb_id not in acl:
        raise AuthError(
            "kb not authorised for this key",
            code=1003,
            # NB: the key's full kb_acl is deliberately NOT echoed back —
            # doing so hands an unauthorised caller an enumeration of every
            # kb_id their (or a stolen) key may reach.
            details={"kb_id": kb_id},
        )


def require_tenant_access(request: Request, tenant_id: str) -> None:
    """Raise ``AuthError(1003)`` when ``tenant_id`` is not the caller's tenant.

    Counterpart to :func:`require_kb_access` for endpoints keyed by tenant
    rather than by kb (e.g. ``GET /v1/audit``). Such endpoints must not let a
    caller pass an arbitrary ``tenant_id`` — otherwise any key can read any
    tenant's rows. Unauthenticated (auth disabled) callers are unrestricted,
    matching :func:`require_kb_access`.
    """
    auth: AuthContext | None = getattr(request.state, "auth", None)
    if auth is None or not auth.authenticated:
        return
    if tenant_id != auth.tenant_id:
        raise AuthError(
            "tenant not authorised for this key",
            code=1003,
            details={"tenant_id": tenant_id},
        )
