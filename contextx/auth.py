"""API authentication — map API keys to a tenant + principals.

The service must NOT trust `tenant_id`/`principals` from the request body (any
caller could then impersonate any tenant). Instead each API key is bound
server-side to exactly one tenant and set of principals; the authenticated
identity is what flows into retrieval, so tenant isolation is real end-to-end.

Keys are stored hashed (SHA-256), never in plaintext; lookups are constant-time.
Load keys from a dict, a JSON file, or the `CONTEXTX_API_KEYS` env var:

    CONTEXTX_API_KEYS='{"sk-acme-123": {"tenant_id": "acme", "principals": ["execs"]}}'
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuthContext:
    tenant_id: str
    principals: list[str] = field(default_factory=list)
    key_id: str = ""


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class APIKeyAuth:
    def __init__(self, keys: dict[str, dict] | None = None) -> None:
        # {hashed_key: AuthContext}
        self._by_hash: dict[str, AuthContext] = {}
        for raw, meta in (keys or {}).items():
            self.add(
                raw,
                tenant_id=meta.get("tenant_id", "default"),
                principals=meta.get("principals", []),
                key_id=meta.get("key_id", ""),
            )

    def add(self, raw_key: str, tenant_id: str,
            principals: list[str] | None = None, key_id: str = "") -> None:
        self._by_hash[_hash(raw_key)] = AuthContext(
            tenant_id=tenant_id, principals=list(principals or []),
            key_id=key_id or tenant_id,
        )

    def authenticate(self, raw_key: str | None) -> AuthContext | None:
        if not raw_key:
            return None
        h = _hash(raw_key)
        for stored, ctx in self._by_hash.items():
            if hmac.compare_digest(stored, h):  # constant-time
                return ctx
        return None

    @property
    def configured(self) -> bool:
        return bool(self._by_hash)

    @classmethod
    def from_env(cls, var: str = "CONTEXTX_API_KEYS") -> "APIKeyAuth":
        raw = os.environ.get(var)
        return cls(json.loads(raw)) if raw else cls()

    @classmethod
    def from_file(cls, path: str) -> "APIKeyAuth":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
