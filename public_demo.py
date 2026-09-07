"""Small, fail-closed settings helpers for the anonymous public demo."""
from __future__ import annotations

import os

LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]", "testserver")
SEARCH_PATHS = frozenset(("/search", "/api/search", "/api/query"))
MAX_QUERY_BYTES = 8_000
MAX_QUERY_STRING_BYTES = 24_000


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean flag")


def allowed_hosts() -> list[str]:
    configured = [host.strip().lower() for host in os.getenv("BAHITH_ALLOWED_HOSTS", "").split(",") if host.strip()]
    if any("*" in host or ":" in host or "/" in host or any(character.isspace() for character in host)
           for host in configured):
        raise ValueError("BAHITH_ALLOWED_HOSTS must contain exact host names without schemes, paths, or wildcards")
    return list(dict.fromkeys((*LOCAL_HOSTS, *configured)))
