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
from utils import odds_budget, odds_store
from utils.odds_fetcher import upcoming_events, _parse_event_odds as parse_quotes
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")
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
_cache_date: str | None = None
_cache_ts: float = 0.0
_requests_remaining: Optional[int] = None


def get_live_wnba_odds(force_refresh: bool = False, target_date=None, max_events: int = 2) -> dict:
    """Cache-only reads; explicit refresh buys 3 market credits per event, up to max_events."""
    global _cache, _cache_ts, _cache_date
    target = target_date or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    fallback = _cache if _cache_date == target and 0 <= time.time() - _cache_ts < _CACHE_TTL else {}
    if not force_refresh:
        return fallback
    if not API_KEY:
        odds_budget.report('Odds API key is not configured. Add THE_ODDS_API_KEY to refresh.')
        return fallback
    try:
        events = upcoming_events(_fetch_events(), target)[:max_events]
        if not events:
            odds_budget.report('No upcoming games on the selected Eastern date.')
            return fallback
        out, successes = {}, 0
        for event in events:
            data = _fetch_event_odds(event['id'], 'player_points,player_rebounds,player_assists')
            if data is not None:
                successes += 1
                _parse_event_odds(dict(event, **data), out, target_date=target)
        if successes:
            _cache, _cache_ts, _cache_date = out, time.time(), target
            odds_store.save('wnba', out)
            return out
    except Exception:
        logger.warning('WNBA odds refresh failed; only fresh same-slate cache may be used')
    return fallback


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


def _filter_tonight_events(events):
    return upcoming_events(events)


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


_TEAM_NAMES = {
    'Atlanta Dream': 'ATL', 'Chicago Sky': 'CHI', 'Connecticut Sun': 'CON',
    'Dallas Wings': 'DAL', 'Golden State Valkyries': 'GSV', 'Indiana Fever': 'IND',
    'Las Vegas Aces': 'LVA', 'Los Angeles Sparks': 'LAS', 'Minnesota Lynx': 'MIN',
    'New York Liberty': 'NYL', 'Phoenix Mercury': 'PHX', 'Seattle Storm': 'SEA',
    'Washington Mystics': 'WAS', 'Portland Fire': 'PDX', 'Toronto Tempo': 'TOR',
}


def _parse_event_odds(event_data, out, target_date=None):
    return parse_quotes(event_data, out, target_date, _TEAM_NAMES)


def _get(url: str, params: dict) -> Optional[requests.Response]:
    return odds_budget.request(requests.get, url, params)


def _track_quota(resp: requests.Response) -> None:
    global _requests_remaining
    val = resp.headers.get("x-requests-remaining")
    if val:
        try:
            _requests_remaining = int(val)
        except ValueError:
            pass
