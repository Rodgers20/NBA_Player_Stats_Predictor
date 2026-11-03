"""On-disk payload cache shared by the keyed source adapters.

Both keyed sources are rate limited — football-data.org at 10 requests/minute,
API-Football at 100 requests/day — so responses are cached to the raw layer
rather than re-requested. The rule that matters is that **"cached" must mean
"recent"**: an earlier version of this logic reused any cache file that existed,
with no expiry at all, which froze a live season permanently on its first fetch.

This lives in one place because the two adapters had independently grown
slightly different versions of it, and the differences were bugs rather than
intent — one handled a corrupted cache file, the other raised straight out of
`json.loads`.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from ..storage import atomic_path

log = logging.getLogger(__name__)


def is_fresh(path: Path, ttl: float) -> bool:
    """True when `path` exists, is non-empty and is younger than `ttl` seconds.

    A non-positive `ttl` always reports stale, which is how callers force a
    refetch.
    """
    if ttl <= 0:
        return False
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    if stat.st_size == 0:
        return False
    return (time.time() - stat.st_mtime) < ttl


def read_json(path: Path) -> Any | None:
    """Cached payload, or None if it is missing, empty or unreadable.

    A truncated cache file — a crash mid-write on a volume without atomic
    replace, say — must degrade to "no cache" rather than taking the caller
    down.
    """
    try:
        raw = path.read_text()
    except (FileNotFoundError, OSError):
        return None
    if not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("discarding unreadable cache %s", path)
        return None


def write_json(path: Path, payload: Any) -> None:
    """Publish a payload atomically, so a partial write is never readable."""
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(payload))
