"""MinIOObjectStore (11-plugins-builtin.md §11.2.4, full profile).

S3-compatible object storage for binary atom payloads (images, formula
rasters). Key convention matches :class:`LocalFSObjectStore`:
``f"{kb_id}/{doc_id}/{atom_id}.bin"``.

Not part of the SPI seven interfaces; it is an ingestion dependency injected
into plugins that produce ``payload_ref`` atoms (13-parsing.md), parallel to
the LocalFS store. The ``aiobotocore`` async client is used for all I/O.

Third-party exceptions (``botocore.*`` / ``aiobotocore.*``) are translated at
the plugin boundary (02-core.md §2.3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ragx.core.exceptions import (
    PluginContractError,
    PluginTimeoutError,
    StoreUnavailableError,
)

_NAME = "minio"
logger = logging.getLogger("ragx.plugins.object_store_minio")


class MinIOObjectStore:
    """S3-compatible object store backed by MinIO (or any S3 endpoint)."""

    name: str = _NAME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.endpoint: str = str(cfg.get("endpoint", "http://localhost:9000"))
        self.bucket: str = str(cfg.get("bucket", "ragx"))
        self.auth_env: str = str(cfg.get("auth_env", "MINIO_AUTH"))
        self.region: str = str(cfg.get("region", "us-east-1"))
        self._session: Any = None
        self._client: Any = None
        self._client_ctx: Any = None

    # -- client lifecycle ---------------------------------------------------
    def _resolve_auth(self) -> tuple[str | None, str | None]:
        """Parse ``auth_env`` → (access_key, secret_key); fall back to AWS env."""
        raw = os.environ.get(self.auth_env, "")
        if raw and ":" in raw:
            access, secret = raw.split(":", 1)
            return access, secret
        # Fall back to the standard AWS env vars (boto3 convention).
        access = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        return (access or None, secret or None)

    async def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from aiobotocore.session import get_session  # type: ignore[import-not-found]
        except ImportError as exc:
            raise PluginContractError(
                "aiobotocore package not installed (pip install ragx[minio])",
                details={"error": str(exc)},
            ) from exc

        self._session = get_session()
        access, secret = self._resolve_auth()
        kwargs: dict[str, Any] = {
            "endpoint_url": self.endpoint,
            "region_name": self.region,
        }
        if access and secret:
            kwargs["aws_access_key_id"] = access
            kwargs["aws_secret_access_key"] = secret
        self._client_ctx = self._session.create_client("s3", **kwargs)
        self._client = await self._client_ctx.__aenter__()
        return self._client

    async def _ensure_bucket(self) -> None:
        client = await self._ensure_client()
        try:
            await client.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "404" in exc_str or "notfound" in exc_str or "nosuchbucket" in exc_str:
                try:
                    await client.create_bucket(Bucket=self.bucket)
                except Exception as create_exc:
                    if "bucketalreadyownedbyyou" not in str(create_exc).lower() and \
                       "bucketalreadyexists" not in str(create_exc).lower():
                        raise self._translate(create_exc, "create bucket") from create_exc
            elif "timeout" in exc_str:
                raise PluginTimeoutError(
                    "MinIO bucket check timed out", details={"error": str(exc)}
                ) from exc
            else:
                # Connection errors: translate to StoreUnavailableError
                raise self._translate(exc, "head bucket") from exc

    # -- public API (mirrors LocalFSObjectStore) ----------------------------
    async def put(self, key: str, data: bytes) -> str:
        client = await self._ensure_client()
        try:
            await client.put_object(
                Bucket=self.bucket, Key=key, Body=data,
                ContentLength=len(data),
            )
            return key
        except Exception as exc:
            raise self._translate(exc, "put", key=key) from exc

    async def get(self, key: str) -> bytes | None:
        client = await self._ensure_client()
        try:
            response = await client.get_object(Bucket=self.bucket, Key=key)
            async with response["Body"] as stream:
                return await stream.read()
        except Exception as exc:
            exc_str = str(exc).lower()
            if "nosuchkey" in exc_str or "404" in exc_str or "notfound" in exc_str:
                return None
            raise self._translate(exc, "get", key=key) from exc

    async def delete(self, key: str) -> None:
        client = await self._ensure_client()
        try:
            await client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "nosuchkey" in exc_str or "404" in exc_str:
                return  # deleting a non-existent key is a no-op
            raise self._translate(exc, "delete", key=key) from exc

    async def exists(self, key: str) -> bool:
        client = await self._ensure_client()
        try:
            await client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            exc_str = str(exc).lower()
            if "nosuchkey" in exc_str or "404" in exc_str or "notfound" in exc_str:
                return False
            raise self._translate(exc, "exists", key=key) from exc

    async def list_prefix(self, prefix: str) -> list[str]:
        client = await self._ensure_client()
        try:
            paginator = client.get_paginator("list_objects_v2")
            keys: list[str] = []
            async for page in paginator.paginate(
                Bucket=self.bucket, Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    keys.append(str(obj["Key"]))
            return sorted(keys)
        except Exception as exc:
            raise self._translate(exc, "list_prefix") from exc

    async def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix; returns the count deleted."""
        keys = await self.list_prefix(prefix)
        if not keys:
            return 0
        client = await self._ensure_client()
        deleted = 0
        # S3 batch delete: max 1000 per request.
        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            try:
                await client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
                )
                deleted += len(batch)
            except Exception as exc:
                raise self._translate(exc, "delete_prefix") from exc
        return deleted

    # -- exception translation ----------------------------------------------
    @staticmethod
    def _translate(exc: Exception, context: str, *, key: str = "") -> Exception:
        exc_str = str(exc).lower()
        details: dict[str, Any] = {"error": str(exc)}
        if key:
            details["key"] = key
        if "timeout" in exc_str or "timed out" in exc_str or "readtimeout" in exc_str:
            return PluginTimeoutError(
                f"MinIO {context} timed out", details=details
            )
        if ("connection" in exc_str or "endpoint" in exc_str
                or "refused" in exc_str or "unreachable" in exc_str
                or "proxy" in exc_str):
            return StoreUnavailableError(
                f"MinIO {context}: store unreachable", code=9001, details=details
            )
        if "accessdenied" in exc_str or "forbidden" in exc_str or "signature" in exc_str:
            return PluginContractError(
                f"MinIO {context}: access denied", details=details
            )
        return StoreUnavailableError(
            f"MinIO {context} failed", code=9001, details=details
        )

    # -- lifecycle ----------------------------------------------------------
    async def startup(self) -> None:
        await self._ensure_client()
        await self._ensure_bucket()

    async def shutdown(self) -> None:
        if self._client_ctx is not None:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_ctx = None
            self._client = None

    async def ping(self) -> bool:
        """Health check (used by /v1/health?deep=true)."""
        try:
            client = await self._ensure_client()
            await client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False
