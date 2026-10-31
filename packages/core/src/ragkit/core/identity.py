"""Who is making a request.

The server sets the caller's identity for the duration of a request; searches
made anywhere inside it (including by agents' tools) filter by its tenant and
principals unless told otherwise.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from ragkit.core.config import ALL_SCOPES


@dataclass(frozen=True)
class Identity:
    tenant_id: str = "default"
    user_id: str | None = None
    principals: tuple[str, ...] = ()
    scopes: frozenset[str] = field(default_factory=lambda: frozenset(ALL_SCOPES))

    def can(self, scope: str) -> bool:
        return scope in self.scopes or "admin" in self.scopes


ANONYMOUS = Identity()

current_identity: ContextVar[Identity | None] = ContextVar("ragkit_identity", default=None)


def get_identity() -> Identity:
    return current_identity.get() or ANONYMOUS


@contextmanager
def acting_as(identity: Identity) -> Iterator[Identity]:
    token = current_identity.set(identity)
    try:
        yield identity
    finally:
        current_identity.reset(token)
