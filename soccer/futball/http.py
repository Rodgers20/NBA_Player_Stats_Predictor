"""Polite, cached HTTP fetching.

Two rules drive this module:

1. **Be a good citizen.** Every source here is free and community-supported.
   We rate-limit, back off on failure, and never hammer. FBref lost its Opta
   feed in Jan 2026 because of scraping at scale; that is a cautionary tale we
   design around rather than repeat.
2. **Raw bytes are immutable.** A fetched payload is written once, verified by
   SHA-256, and never edited. Every downstream number can therefore be traced
   back to the exact bytes it came from.
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "futball-dashboard/0.1 (personal analytics project)"


@dataclass(frozen=True)
class FetchResult:
    path: Path
    sha256: str
    bytes_written: int
    from_cache: bool


class PoliteClient:
    """Rate-limited HTTP client that persists responses to the raw layer."""

    def __init__(
        self,
        *,
        min_interval: float = 0.25,
        timeout: float = 60.0,
        max_retries: int = 4,
        follow_redirects: bool = True,
    ) -> None:
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._last_request = 0.0
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=follow_redirects,
        )

    def __enter__(self) -> "PoliteClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def get_bytes(self, url: str) -> bytes:
        """GET with retry + exponential backoff. Raises on final failure."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = self._client.get(url)
                if resp.status_code == 404:
                    raise FileNotFoundError(f"404 Not Found: {url}")
                # football-data.co.uk answers a not-yet-published season file with
                # 300 Multiple Choices (Apache content negotiation offering
                # near-miss names), not 404. It means "absent", so do not retry.
                if resp.status_code in (300, 301, 302, 303, 307, 308) and not resp.has_redirect_location:
                    raise FileNotFoundError(f"{resp.status_code} not available: {url}")
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 2**attempt))
                    log.warning("429 from %s, sleeping %.1fs", url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.content
            except FileNotFoundError:
                raise
            except Exception as exc:  # noqa: BLE001 - retried below
                last_exc = exc
                wait = 2**attempt
                log.warning(
                    "fetch failed (%s/%s) %s: %s — retrying in %ss",
                    attempt + 1, self._max_retries, url, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"giving up on {url}") from last_exc

    def fetch_to(self, url: str, dest: Path, *, skip_if_exists: bool = True) -> FetchResult:
        """Fetch `url` into `dest`, writing atomically. Resumable by design."""
        if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
            data = dest.read_bytes()
            return FetchResult(dest, hashlib.sha256(data).hexdigest(), len(data), True)

        data = self.get_bytes(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: a partial download must never look like a complete one.
        tmp = dest.with_suffix(dest.suffix + ".partial")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return FetchResult(dest, hashlib.sha256(data).hexdigest(), len(data), False)
