"""Authentication, scopes and rate limits for the HTTP API.

Modes (under `auth` in ragkit.yaml):

- `none`: every caller is the default tenant with all scopes (local use).
- `api_key`: `Authorization: Bearer <key>` or `X-API-Key: <key>`. Each key is
  read from an environment variable and maps to a tenant, principals and scopes.
- `jwt`: a bearer token signed with a shared secret (HS256) or a key from a
  JWKS URL (RS256/ES256), with tenant, principals and scopes read from claims.

The caller's identity is set for the whole request, so searches (including
those made by agent tools) only see the caller's tenant and readable documents.
Conversation threads are scoped by tenant and user, so one caller cannot
continue or approve another caller's thread.
"""

import asyncio
import hmac
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

import jwt
from fastapi import HTTPException, Request

from ragkit.core.config import ALL_SCOPES, AuthConfig
from ragkit.core.errors import ConfigError
from ragkit.core.identity import ANONYMOUS, Identity

Authenticate = Callable[[Request], Awaitable[Identity]]


def build_authenticator(config: AuthConfig) -> Authenticate:
    if config.mode == "none":

        async def anonymous(_request: Request) -> Identity:
            return ANONYMOUS

        return anonymous
    if config.mode == "api_key":
        return _api_keys(config)
    return _jwt(config)


def _api_keys(config: AuthConfig) -> Authenticate:
    keys: list[tuple[bytes, Identity]] = []
    for entry in config.api_keys:
        secret = os.environ.get(entry.key_env)
        if not secret:
            raise ConfigError(f"API key variable {entry.key_env} is not set.")
        identity = Identity(
            tenant_id=entry.tenant,
            user_id=entry.name or entry.key_env.lower(),
            principals=tuple(entry.principals),
            scopes=frozenset(entry.scopes),
        )
        keys.append((secret.encode(), identity))

    async def authenticate(request: Request) -> Identity:
        presented = _bearer(request) or request.headers.get("x-api-key")
        if not presented:
            raise _unauthorized("Missing API key.")
        candidate = presented.encode()
        # Compare against every key so timing does not reveal which one matched.
        matches = [identity for secret, identity in keys if hmac.compare_digest(secret, candidate)]
        if not matches:
            raise _unauthorized("Invalid API key.")
        return matches[0]

    return authenticate


def _jwt(config: AuthConfig) -> Authenticate:
    settings = config.jwt
    assert settings is not None
    secret = None
    jwks = None
    if settings.secret_env:
        secret = os.environ.get(settings.secret_env)
        if not secret:
            raise ConfigError(f"JWT secret variable {settings.secret_env} is not set.")
    else:
        jwks = jwt.PyJWKClient(settings.jwks_url, cache_keys=True, lifespan=600)

    async def authenticate(request: Request) -> Identity:
        token = _bearer(request)
        if not token:
            raise _unauthorized("Missing bearer token.")
        try:
            key: Any = secret
            if jwks is not None:
                key = (await asyncio.to_thread(jwks.get_signing_key_from_jwt, token)).key
            claims = jwt.decode(
                token,
                key,
                algorithms=settings.algorithms,
                issuer=settings.issuer,
                audience=settings.audience,
                options={"require": ["exp"], "verify_aud": settings.audience is not None},
            )
        except jwt.PyJWTError as exc:
            raise _unauthorized(f"Invalid token: {exc}") from exc
        scopes = claims.get(settings.scopes_claim)
        if isinstance(scopes, str):
            scopes = scopes.split()
        granted = [s for s in (scopes or settings.default_scopes) if s in ALL_SCOPES]
        principals = claims.get(settings.principals_claim) or []
        if isinstance(principals, str):
            principals = [principals]
        return Identity(
            tenant_id=str(claims.get(settings.tenant_claim) or "default"),
            user_id=str(claims.get("sub")) if claims.get("sub") else None,
            principals=tuple(str(p) for p in principals),
            scopes=frozenset(granted),
        )

    return authenticate


class RateLimiter:
    """Token bucket per caller, refilled continuously. Limits are per server process."""

    def __init__(self, per_minute: int | None) -> None:
        self.per_minute = per_minute
        self._buckets: dict[tuple[str, str | None], tuple[float, float]] = {}

    def check(self, identity: Identity) -> None:
        if not self.per_minute:
            return
        key = (identity.tenant_id, identity.user_id)
        now = time.monotonic()
        tokens, updated = self._buckets.get(key, (float(self.per_minute), now))
        tokens = min(float(self.per_minute), tokens + (now - updated) * self.per_minute / 60)
        if tokens < 1:
            retry = max(1, int((1 - tokens) * 60 / self.per_minute) + 1)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"Retry-After": str(retry)},
            )
        self._buckets[key] = (tokens - 1, now)


def scoped_thread(identity: Identity, thread_id: str) -> str:
    """The thread id used internally, so threads never collide across callers."""
    return f"{identity.tenant_id}/{identity.user_id or '-'}/{thread_id}"


def require(identity: Identity, scope: str) -> None:
    if not identity.can(scope):
        raise HTTPException(status_code=403, detail=f"This key or token lacks the {scope} scope.")


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})
