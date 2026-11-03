"""HTTP for the free NHL web API: shared session (follows redirects), retry with backoff, polite pacing."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRIES = 4
BACKOFF_SECONDS = 1.5
PAUSE_SECONDS = 0.05
MAX_WORKERS = 6
_local = threading.local()


class FeedError(RuntimeError):
    """The NHL API did not answer, or answered in a shape we do not know."""


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "nhlpred/1.0 (personal research)"
    return _local.s


def get_json(url: str, *, missing_ok: bool = False) -> dict[str, Any] | None:
    """GET JSON; 404 returns None when ``missing_ok``."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        time.sleep(PAUSE_SECONDS)
        try:
            r = _session().get(url, timeout=30, allow_redirects=True)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404 and missing_ok:
                return None
            last = FeedError(f"NHL HTTP {r.status_code} for {url}")
        except (requests.RequestException, ValueError) as exc:
            last = exc
        time.sleep(BACKOFF_SECONDS * 2**attempt)
    raise FeedError(str(last))
