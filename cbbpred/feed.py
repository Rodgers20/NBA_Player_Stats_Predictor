"""HTTP for the free ESPN endpoints: one shared client, retry with backoff, polite request pacing."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(30.0)
RETRIES = 4
BACKOFF_SECONDS = 1.5      # first retry waits this long, then doubles
PAUSE_SECONDS = 0.05       # gap before each request per worker thread; keeps 6-8 workers polite
MAX_WORKERS = 6
LIMITS = httpx.Limits(max_connections=MAX_WORKERS * 2, max_keepalive_connections=MAX_WORKERS)

_lock = threading.Lock()
_client: httpx.Client | None = None


class FeedError(RuntimeError):
    """ESPN did not answer, or answered in a shape we do not know."""


def client() -> httpx.Client:
    global _client
    with _lock:
        if _client is None:
            _client = httpx.Client(timeout=TIMEOUT, limits=LIMITS, headers={"Accept-Encoding": "gzip"})
        return _client


def get_json(url: str, *, missing_ok: bool = False, **params: Any) -> dict[str, Any] | None:
    """GET ``url`` as JSON. 404 returns None when ``missing_ok`` (an event with no odds page), else raises."""
    last: Exception | None = None
    for attempt in range(RETRIES):
        time.sleep(PAUSE_SECONDS)
        try:
            resp = client().get(url, params=params)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404 and missing_ok:
                return None
            last = FeedError(f"ESPN HTTP {resp.status_code} for {url} {params}")
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
        time.sleep(BACKOFF_SECONDS * 2**attempt)
    raise FeedError(str(last))
