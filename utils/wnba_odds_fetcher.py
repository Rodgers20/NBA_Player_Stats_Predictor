"""WNBA player prop odds via The Odds API (sport = basketball_wnba).

Independent of utils/odds_fetcher.py (NBA) so both can evolve separately.
Returns odds in the same shape so downstream props scoring code doesn't care.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("THE_ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_wnba"

PREFERRED_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]

MARKET_TO_STAT = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "FG3M",
    "player_points_rebounds": "PTS+REB",
    "player_points_assists": "PTS+AST",
    "player_rebounds_assists": "REB+AST",
    "player_points_rebounds_assists": "PTS+REB+AST",
}

_ALL_MARKETS = ",".join(MARKET_TO_STAT.keys())

_CACHE_TTL = 30 * 60   # 30 min — conserves Odds API quota

# ── In-memory cache ─────────────────────────────────────────────────────────
_cache: dict = {}         # {player_name: {stat: odds_dict}}
_cache_ts: float = 0.0
_requests_remaining: Optional[int] = None


def get_live_wnba_odds(force_refresh: bool = False) -> dict:
    """Return live WNBA player prop odds — only for events happening tonight (US ET).

    Returns nested dict:
        {player_name: {stat: {"line", "over_price", "under_price", "bookmaker"}}}

    On failure, returns the last-good cache rather than an empty dict.
    """
    global _cache, _cache_ts

    if not API_KEY:
        logger.debug("[WNBA-Odds] THE_ODDS_API_KEY not set")
        return _cache

    if not force_refresh and _cache and (time.time() - _cache_ts) < _CACHE_TTL:
        return _cache

    try:
        events = _fetch_events()
        if not events:
            logger.info("[WNBA-Odds] No WNBA events returned")
            return _cache

        # Filter to events happening in today's ET window (12:00 ET today → 06:00 ET tomorrow)
        tonight_events = _filter_tonight_events(events)
        if not tonight_events:
            logger.info("[WNBA-Odds] No WNBA events tonight (all future or past)")
            return _cache

        out: dict = {}
        successes = 0
        failures = 0
        for e in tonight_events:
            data = _fetch_event_odds(e["id"], _ALL_MARKETS)
            if data is None:
                failures += 1
                continue
            successes += 1
            _parse_event_odds(data, out)

        # Guard: if every event 401'd (quota exhausted), keep the previous cache
        if successes == 0 and _cache:
            logger.warning(f"[WNBA-Odds] All {failures} event fetches failed — keeping last cache ({len(_cache)} players)")
            return _cache

        _cache = out
        _cache_ts = time.time()
        logger.info(f"[WNBA-Odds] Loaded {len(out)} players from {successes}/{len(tonight_events)} tonight events")
        return out
    except Exception as e:
        logger.warning(f"[WNBA-Odds] fetch failed: {e}")
        return _cache


def get_wnba_player_odds(player_name: str, stat: str) -> Optional[dict]:
    """Convenience: get one player's odds for one stat, or None."""
    odds = get_live_wnba_odds()
    return odds.get(player_name, {}).get(stat)


def get_wnba_requests_remaining() -> Optional[int]:
    return _requests_remaining


# ── Internals ────────────────────────────────────────────────────────────────

def _fetch_events() -> list[dict]:
    """Return raw event dicts (id + commence_time), or [] on failure."""
    url = f"{BASE_URL}/sports/{SPORT}/events"
    resp = _get(url, {"apiKey": API_KEY, "dateFormat": "iso"})
    if resp is None:
        return []
    return resp.json()


def _fetch_event_ids() -> list[str]:
    """Legacy helper kept for tests. Prefer _fetch_events + _filter_tonight_events."""
    return [e["id"] for e in _fetch_events()]


def _filter_tonight_events(events: list[dict]) -> list[dict]:
    """Return events whose tip-off falls in the current WNBA game night window.

    WNBA "game night" in US ET is approximately noon to 6am next-day (covers
    late-tip West Coast games). All commence_time values are UTC ISO 8601;
    ET is UTC-4 in summer (DST). We use a fixed 4-hour offset (WNBA season
    is entirely within DST, May-Oct).
    """
    et_offset = timedelta(hours=-4)
    now_et = datetime.now(timezone.utc) + et_offset
    # Window: noon today ET → 6am tomorrow ET
    window_start = now_et.replace(hour=12, minute=0, second=0, microsecond=0)
    if now_et.hour < 6:
        # Early morning — we're still on "last night's" game slate
        window_start -= timedelta(days=1)
    window_end = window_start + timedelta(hours=18)   # noon → 6am next day

    tonight = []
    for e in events:
        try:
            tip_utc = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            tip_et = tip_utc + et_offset
            if window_start <= tip_et <= window_end:
                tonight.append(e)
        except Exception:
            continue
    return tonight


def _fetch_event_odds(event_id: str, markets: str) -> Optional[dict]:
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
    }
    resp = _get(url, params)
    if resp is None:
        return None
    _track_quota(resp)
    return resp.json()


def _parse_event_odds(event_data: dict, out: dict) -> None:
    bookmakers = event_data.get("bookmakers", [])

    def _rank(b):
        k = b.get("key", "")
        return PREFERRED_BOOKS.index(k) if k in PREFERRED_BOOKS else 99

    for book in sorted(bookmakers, key=_rank):
        book_name = book.get("title", book.get("key", "Unknown"))
        for market in book.get("markets", []):
            stat = MARKET_TO_STAT.get(market.get("key", ""))
            if not stat:
                continue
            by_player: dict = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                if not player:
                    continue
                by_player.setdefault(player, {})[outcome.get("name", "")] = {
                    "price": outcome.get("price"),
                    "point": outcome.get("point"),
                }
            for player, sides in by_player.items():
                over = sides.get("Over", {})
                under = sides.get("Under", {})
                line = over.get("point") or under.get("point")
                if line is None:
                    continue
                player_dict = out.setdefault(player, {})
                if stat not in player_dict:
                    player_dict[stat] = {
                        "line": float(line),
                        "over_price": over.get("price"),
                        "under_price": under.get("price"),
                        "bookmaker": book_name,
                    }


def _get(url: str, params: dict) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp
    except Exception as e:
        logger.warning(f"[WNBA-Odds] GET {url} failed: {e}")
        return None


def _track_quota(resp: requests.Response) -> None:
    global _requests_remaining
    val = resp.headers.get("x-requests-remaining")
    if val:
        try:
            _requests_remaining = int(val)
        except ValueError:
            pass
