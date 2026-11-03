"""Football news from public RSS feeds.

Two publishers with free, stable football feeds. No key, no quota, and no
scraping — an RSS feed is published precisely so it can be read this way.

**Headline, source, timestamp and link only.** The article body is never stored
or shown. A feed exists to point at the publisher's page, and reproducing the
text would be taking something the feed was not offering.

Parsing uses the standard library rather than a dependency: these are two
well-formed feeds, not the open web.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .. import config
from ..http import PoliteClient
from . import _cache

log = logging.getLogger(__name__)

RAW_NEWS = config.RAW / "news"

# Half an hour is frequent enough for a dashboard and gentle on the publishers.
NEWS_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class Feed:
    key: str
    name: str
    url: str


FEEDS: tuple[Feed, ...] = (
    Feed("bbc", "BBC Sport", "https://feeds.bbci.co.uk/sport/football/rss.xml"),
    Feed("guardian", "The Guardian", "https://www.theguardian.com/football/rss"),
)

_TAG = re.compile(r"<[^>]+>")

# Phrases that contain a club's name as a whole word but mean a different club.
# Word boundaries cannot separate these; only naming them can.
_NOT_THIS_CLUB: dict[str, tuple[str, ...]] = {
    "Inter": ("Inter Miami",),
    "Real Madrid": (),
    "Athletic Club": ("Athletic Bilbao",),  # the same club, so never an exclusion
}


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(_TAG.sub(" ", node.text).split())


def _published(item: ET.Element) -> str | None:
    raw = _text(item.find("pubDate"))
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def parse_feed(xml: str, feed: Feed) -> list[dict]:
    """Headlines from one RSS document."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        log.warning("news feed %s did not parse: %s", feed.key, exc)
        return []

    items: list[dict] = []
    for item in root.iter("item"):
        title = _text(item.find("title"))
        link = _text(item.find("link"))
        if not title or not link:
            continue
        items.append({
            "title": title,
            "link": link,
            "source": feed.name,
            "source_key": feed.key,
            "published": _published(item),
            # A one-line standfirst is part of the feed's own summary; the
            # article body is not fetched.
            "summary": _text(item.find("description"))[:280] or None,
        })
    return items


def _cache_path(feed: Feed):
    return RAW_NEWS / f"{feed.key}.xml"


def fetch_feed(feed: Feed, *, ttl: float = NEWS_TTL_SECONDS) -> list[dict]:
    """One feed, cached to the raw layer with a TTL."""
    dest = _cache_path(feed)
    if _cache.is_fresh(dest, ttl) and dest.exists():
        return parse_feed(dest.read_text(encoding="utf-8", errors="replace"), feed)

    try:
        with PoliteClient(min_interval=1.0, max_retries=2) as client:
            data = client.get_bytes(feed.url)
    except Exception as exc:  # noqa: BLE001 - stale headlines beat none
        log.warning("news feed %s failed: %s", feed.key, exc, exc_info=True)
        if dest.exists():
            return parse_feed(dest.read_text(encoding="utf-8", errors="replace"), feed)
        return []

    dest.parent.mkdir(parents=True, exist_ok=True)
    text = data.decode("utf-8", errors="replace")
    dest.write_text(text, encoding="utf-8")
    return parse_feed(text, feed)


def headlines(limit: int = 30, *, ttl: float = NEWS_TTL_SECONDS) -> list[dict]:
    """Everything from every feed, newest first."""
    items: list[dict] = []
    for feed in FEEDS:
        items.extend(fetch_feed(feed, ttl=ttl))
    items.sort(key=lambda i: i.get("published") or "", reverse=True)
    return items[:limit]


def for_competition(code: str, limit: int = 20,
                    *, ttl: float = NEWS_TTL_SECONDS) -> dict:
    """Headlines mentioning a club in this competition.

    Matching is on **club names from the directory**, not on the competition
    name — a story about Arsenal rarely says "Premier League" in its headline.
    That makes this a filter rather than a guarantee, so the response says so
    and the tab is labelled as football news rather than league news.

    Two guards, because a plain substring test attributed an MLS story to Inter:

    * **Word boundaries** stop a name matching inside a longer word, which is
      what claimed "interview" and "international" for Inter.
    * **An exclusion list** handles the case boundaries cannot: "Inter Miami"
      contains the standalone word "Inter", so the club name alone is genuinely
      ambiguous and only knowing the other club resolves it.
    """
    from .. import clubs

    everything = headlines(limit=200, ttl=ttl)

    # Longest spelling first, so "Manchester City" is tried before "Man City"
    # and the more specific attribution wins.
    wanted: list[tuple[re.Pattern[str], str]] = []
    for name, entry in clubs.directory().items():
        if code not in entry.get("competitions", ()):
            continue
        display = entry.get("short_name") or name
        for spelling in sorted(
            {s for s in (entry.get("short_name"), name) if s and len(s) >= 4},
            key=len, reverse=True,
        ):
            wanted.append(
                (re.compile(rf"(?<!\w){re.escape(spelling)}(?!\w)", re.IGNORECASE), display)
            )

    matched: list[dict] = []
    for item in everything:
        haystack = f"{item['title']} {item.get('summary') or ''}"
        hit = None
        for pattern, club in wanted:
            if not pattern.search(haystack):
                continue
            excluded = _NOT_THIS_CLUB.get(club, ())
            if any(phrase.lower() in haystack.lower() for phrase in excluded):
                continue
            hit = club
            break
        if hit:
            matched.append({**item, "club": hit})

    return {
        "competition_code": code,
        "matched": matched[:limit],
        "recent": everything[:limit],
        "note": (
            "Headlines are matched to this competition by club name, so the "
            "list is indicative rather than complete. Titles link to the "
            "publisher; no article text is stored here."
        ),
        "sources": [f.name for f in FEEDS],
    }
