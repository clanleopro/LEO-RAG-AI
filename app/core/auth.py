# app/core/auth.py
"""
Authentication and authorization for LEO Rigging AI.

Implements configurable bearer-token / API-key authentication.
Credentials are compared with constant-time comparison to prevent
timing-based attacks. Credentials are never logged.

Roles / scopes:
  query              – POST /api/query, POST /api/search
  documents:read     – GET /api/pdfs, GET /api/pdfs/{doc_id}
  documents:write    – POST /api/upload, POST /api/ingest
  documents:delete   – DELETE /api/pdfs/{doc_id}
  admin              – GET /info, GET /routes, GET /api/jobs, admin ops
                       (admin implicitly includes all other scopes)
"""
from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger(__name__)

# ─── Role hierarchy ──────────────────────────────────────────
ROLE_SCOPES: dict[str, set[str]] = {
    "query": {"query"},
    "documents:read": {"documents:read"},
    "documents:write": {"documents:write", "documents:read"},
    "documents:delete": {"documents:delete", "documents:read"},
    "admin": {
        "query",
        "documents:read",
        "documents:write",
        "documents:delete",
        "admin",
    },
}


def _resolve_scopes(role: str) -> set[str]:
    return ROLE_SCOPES.get(role, {role})


# ─── Extractors — try Authorization header first, then X-API-Key ─
_bearer_scheme = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _extract_credential(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    api_key: Optional[str] = Security(_api_key_header),
) -> Optional[str]:
    """
    Extract the raw credential from the request.
    Priority: Authorization: Bearer <token>  >  X-API-Key: <key>
    Returns None if neither is present.
    Never logs the credential value.
    """
    if bearer and bearer.credentials:
        return bearer.credentials
    if api_key:
        return api_key
    return None


# ─── Core resolver ───────────────────────────────────────────

def _get_key_roles() -> dict[str, str]:
    """
    Load key→role mapping from settings.
    Imported lazily to avoid circular import during module init.
    """
    from app.services.config import ENV  # noqa: PLC0415
    return ENV.api_key_roles()


def _authenticate(credential: Optional[str], required_scope: str) -> str:
    """
    Validate the credential and ensure it has the required scope.

    Returns the role string on success.
    Raises HTTPException on failure.
    """
    from app.services.config import ENV  # noqa: PLC0415

    # If auth is disabled (dev mode), allow through but log a warning.
    if not ENV.AUTH_ENABLED:
        log.warning(
            "AUTH_ENABLED=false — authentication is disabled. "
            "This MUST NOT be used in production."
        )
        return "admin"

    if not credential:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials. "
                   "Provide a Bearer token in the Authorization header "
                   "or an API key in the X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_roles = _get_key_roles()

    # Constant-time comparison across all keys to prevent timing attacks.
    matched_role: Optional[str] = None
    for stored_key, role in key_roles.items():
        if hmac.compare_digest(stored_key.encode(), credential.encode()):
            matched_role = role
            break

    if matched_role is None:
        # Log failure WITHOUT logging the credential value.
        log.warning("Authentication failed: invalid credential (credential not logged).")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check scope
    granted_scopes = _resolve_scopes(matched_role)
    if required_scope not in granted_scopes:
        log.warning(
            "Authorization denied: required=%s, granted=%s",
            required_scope,
            granted_scopes,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions. Required scope: '{required_scope}'.",
        )

    return matched_role


# ─── FastAPI dependency factories ────────────────────────────

def _make_dep(scope: str):
    """Return a FastAPI dependency that enforces the given scope."""

    def _dep(credential: Optional[str] = Depends(_extract_credential)) -> str:
        return _authenticate(credential, scope)

    _dep.__name__ = f"require_{scope.replace(':', '_')}"
    return _dep


require_query = _make_dep("query")
require_documents_read = _make_dep("documents:read")
require_documents_write = _make_dep("documents:write")
require_documents_delete = _make_dep("documents:delete")
require_admin = _make_dep("admin")


__all__ = [
    "require_query",
    "require_documents_read",
    "require_documents_write",
    "require_documents_delete",
    "require_admin",
]
