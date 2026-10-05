# utils/odds_fetcher.py
"""
Live Sportsbook Odds Fetcher
============================
Fetches real-time NBA player prop odds from The Odds API.

Setup:
  1. Sign up at https://the-odds-api.com  (free tier: 500 requests/month)
  2. Add to your .env file:
       THE_ODDS_API_KEY=your_key_here

Player-prop refresh is explicit and limited to two events / three markets.
All requests use the persistent shared credit budget; ordinary reads are cache-only.

Return shape:
  {
    "Jaylen Brown": {
      "PTS": {"line": 26.5, "over_price": -115, "under_price": -105, "bookmaker": "FanDuel"},
      "REB": {"line": 5.5,  "over_price": -110, "under_price": -110, "bookmaker": "FanDuel"},
      ...
    },
    ...
  }
"""

import os
import time
import logging
import requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from utils import odds_budget

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT   = "basketball_nba"

# Preferred bookmakers (first available wins per player/stat)
PREFERRED_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "pointsbet"]

# Odds API market key → our internal stat key
MARKET_TO_STAT = {
    "player_points":           "PTS",
    "player_rebounds":         "REB",
    "player_assists":          "AST",
    "player_threes":           "FG3M",
    "player_points_rebounds":  "PTS+REB",
    "player_points_assists":   "PTS+AST",
    "player_points_rebounds_assists": "PTS+REB+AST",
}

# Cache TTL: 30 minutes
_CACHE_TTL = 30 * 60

# ── In-memory cache ─────────────────────────────────────────────────────────
_cache: dict = {}          # player_name → {stat → odds_dict}
_cache_date: str | None = None
_cache_ts: float = 0.0     # unix timestamp of last fetch
_requests_remaining: int | None = None  # track quota from response headers

# Explicit refresh may use player props if the configured account permits them.
PLAYER_PROPS_ENABLED: bool = True

# Circuit breaker: if the player-props endpoint returns 401 (paid plan required)
# stop making per-event calls for the rest of the session.
_player_props_unavailable: bool = not PLAYER_PROPS_ENABLED

# ── Game odds cache ──────────────────────────────────────────────────────────
_game_odds_cache: dict = {}   # "(away_abbr)@(home_abbr)" → odds dict
_game_odds_ts: float = 0.0

# The Odds API full team names → our ESPN abbreviations
_TEAM_NAME_TO_ABBR: dict = {
    "Atlanta Hawks":          "ATL", "Boston Celtics":        "BOS",
    "Brooklyn Nets":          "BKN", "Charlotte Hornets":     "CHA",
    "Chicago Bulls":          "CHI", "Cleveland Cavaliers":   "CLE",
    "Dallas Mavericks":       "DAL", "Denver Nuggets":        "DEN",
    "Detroit Pistons":        "DET", "Golden State Warriors": "GSW",
    "Houston Rockets":        "HOU", "Indiana Pacers":        "IND",
    "Los Angeles Clippers":   "LAC", "Los Angeles Lakers":    "LAL",
    "Memphis Grizzlies":      "MEM", "Miami Heat":            "MIA",
    "Milwaukee Bucks":        "MIL", "Minnesota Timberwolves":"MIN",
    "New Orleans Pelicans":   "NOP", "New York Knicks":       "NYK",
    "Oklahoma City Thunder":  "OKC", "Orlando Magic":         "ORL",
    "Philadelphia 76ers":     "PHI", "Phoenix Suns":          "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings":      "SAC",
    "San Antonio Spurs":      "SAS", "Toronto Raptors":       "TOR",
    "Utah Jazz":              "UTA", "Washington Wizards":    "WAS",
}


# ── Public API ──────────────────────────────────────────────────────────────

def get_live_odds(force_refresh: bool = False, target_date: str | None = None) -> dict:
    """Read cache by default. Explicit refresh buys at most six market credits."""
    global _cache, _cache_ts, _cache_date
    target = target_date or datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    fallback = _cache if _cache_date == target and 0 <= time.time() - _cache_ts < _CACHE_TTL else {}
    if not force_refresh:
        return fallback
    if not API_KEY:
        odds_budget.report('Odds API key is not configured. Add THE_ODDS_API_KEY to refresh.')
        return fallback
    if _player_props_unavailable:
        return fallback
    try:
        event_ids = _fetch_event_ids(target)[:2]
        if not event_ids:
            odds_budget.report('No upcoming games on the selected Eastern date.')
            return fallback
        fresh, successes = {}, 0
        for event_id in event_ids:
            payload = _fetch_event_odds(event_id, 'player_points,player_rebounds,player_assists')
            if payload:
                successes += 1
                _parse_event_odds(payload, fresh, target_date=target)
        if successes:
            _cache, _cache_ts, _cache_date = fresh, time.time(), target
            return _cache
    except Exception:
        logger.warning('Odds refresh failed; only fresh same-slate cache may be used')
    return fallback


def get_player_odds(player_name: str, stat: str) -> dict | None:
    """
    Convenience wrapper: return odds for one player/stat combination.

    Returns None if no live odds are available for this player/stat.

    Example:
        odds = get_player_odds("Jaylen Brown", "PTS")
        # {"line": 26.5, "over_price": -115, "under_price": -105, "bookmaker": "FanDuel"}
    """
    all_odds = get_live_odds()
    player_odds = all_odds.get(player_name) or all_odds.get(_normalize_name(player_name))
    if not player_odds:
        return None
    return player_odds.get(stat)


def get_requests_remaining() -> int | None:
    """Return the number of The Odds API requests remaining this month."""
    return _requests_remaining


def get_game_odds(force_refresh: bool = False) -> dict:
    """
    Fetch spread, total (over/under), and moneyline odds for today's NBA games.

    Uses a 30-minute in-memory cache. Returns empty dict on API failure.

    Returns:
        {
          "GSW@LAL": {
            "home_team":   "LAL",
            "away_team":   "GSW",
            "spread": {
              "home_line":   -6.5,       # negative = home favored
              "home_price":  -110,
              "away_line":    6.5,
              "away_price":  -110,
            },
            "total": {
              "line":        224.5,
              "over_price":  -110,
              "under_price": -110,
            },
            "h2h": {
              "home_price":  -280,
              "away_price":  +230,
            },
            "bookmaker": "FanDuel",
          },
          ...
        }
    """
    global _game_odds_cache, _game_odds_ts
    if not force_refresh:
        return _game_odds_cache if 0 <= time.time() - _game_odds_ts < _CACHE_TTL else {}

    if not API_KEY:
        logger.debug("[OddsFetcher] THE_ODDS_API_KEY not set — skipping game odds")
        return {}

    if not force_refresh and _game_odds_cache and (time.time() - _game_odds_ts) < _CACHE_TTL:
        return _game_odds_cache

    logger.info("[OddsFetcher] Fetching game odds (spreads + totals + h2h) …")
    try:
        url = f"{BASE_URL}/sports/{SPORT}/odds"
        params = {
            "apiKey":    API_KEY,
            "regions":   "us",
            "markets":   "spreads,totals,h2h",
            "oddsFormat": "american",
        }
        resp = _get(url, params)
        if resp is None:
            return _game_odds_cache

        _track_quota(resp)
        events = resp.json()
        fresh = {}

        for event in events:
            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")
            home_abbr = _TEAM_NAME_TO_ABBR.get(home_name, home_name[:3].upper())
            away_abbr = _TEAM_NAME_TO_ABBR.get(away_name, away_name[:3].upper())
            key = f"{away_abbr}@{home_abbr}"

            game_odds: dict = {
                "home_team": home_abbr,
                "away_team": away_abbr,
                "spread":    None,
                "total":     None,
                "h2h":       None,
                "bookmaker": None,
            }

            bookmakers = event.get("bookmakers", [])

            def _book_rank(b):
                k = b.get("key", "")
                return PREFERRED_BOOKS.index(k) if k in PREFERRED_BOOKS else 99

            for book in sorted(bookmakers, key=_book_rank):
                book_name = book.get("title", book.get("key", "Unknown"))
                for market in book.get("markets", []):
                    mkey = market.get("key", "")
                    outcomes = market.get("outcomes", [])

                    if mkey == "spreads" and game_odds["spread"] is None:
                        home_out = next((o for o in outcomes if o.get("name") == home_name), None)
                        away_out = next((o for o in outcomes if o.get("name") == away_name), None)
                        if home_out and away_out:
                            game_odds["spread"] = {
                                "home_line":  float(home_out.get("point", 0)),
                                "home_price": int(home_out.get("price", -110)),
                                "away_line":  float(away_out.get("point", 0)),
                                "away_price": int(away_out.get("price", -110)),
                            }
                            game_odds["bookmaker"] = book_name

                    elif mkey == "totals" and game_odds["total"] is None:
                        over_out  = next((o for o in outcomes if o.get("name") == "Over"),  None)
                        under_out = next((o for o in outcomes if o.get("name") == "Under"), None)
                        if over_out:
                            game_odds["total"] = {
                                "line":        float(over_out.get("point", 220)),
                                "over_price":  int(over_out.get("price",  -110)),
                                "under_price": int(under_out.get("price", -110)) if under_out else -110,
                            }

                    elif mkey == "h2h" and game_odds["h2h"] is None:
                        home_out = next((o for o in outcomes if o.get("name") == home_name), None)
                        away_out = next((o for o in outcomes if o.get("name") == away_name), None)
                        if home_out and away_out:
                            game_odds["h2h"] = {
                                "home_price": int(home_out.get("price", -110)),
                                "away_price": int(away_out.get("price", +100)),
                            }

            fresh[key] = game_odds

        _game_odds_cache = fresh
        _game_odds_ts = time.time()
        logger.info(f"[OddsFetcher] Cached game odds for {len(fresh)} games")
        return _game_odds_cache

    except Exception as exc:
        logger.warning(f"[OddsFetcher] Failed to fetch game odds: {exc}")
        return _game_odds_cache


def format_american_odds(price: int) -> str:
    """Format American odds with sign: -110 → '-110', +230 → '+230'."""
    if price is None:
        return "N/A"
    return f"+{price}" if price > 0 else str(price)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_event_ids(target_date=None) -> list[str]:
    """Fetch today's NBA event IDs."""
    url = f"{BASE_URL}/sports/{SPORT}/events"
    resp = _get(url, {"apiKey": API_KEY, "dateFormat": "iso"})
    if resp is None:
        return []
    return [e["id"] for e in upcoming_events(resp.json(), target_date)][:2]


def _fetch_event_odds(event_id: str, markets: str) -> dict | None:
    """Fetch player prop odds for a single event."""
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":    API_KEY,
        "regions":   "us",
        "markets":   markets,
        "oddsFormat": "american",
    }
    resp = _get(url, params)
    if resp is None:
        return None

    _track_quota(resp)
    return resp.json()


def _parse_event_odds(event_data: dict, out: dict, target_date=None, team_map=None) -> None:
    """
    Parse one event's odds payload into the flat {player: {stat: odds}} dict.

    We iterate bookmakers in PREFERRED_BOOKS order so FanDuel lines win over
    less-popular books when both carry the same player prop.
    """
    metadata = event_metadata(event_data, target_date, team_map or _TEAM_NAME_TO_ABBR)
    if metadata is None:
        return
    bookmakers: list[dict] = event_data.get("bookmakers", [])

    # Sort by preferred order (books not in the list go last)
    def _book_rank(b):
        k = b.get("key", "")
        return PREFERRED_BOOKS.index(k) if k in PREFERRED_BOOKS else 99

    for book in sorted(bookmakers, key=_book_rank):
        book_name = book.get("title", book.get("key", "Unknown"))
        for market in book.get("markets", []):
            stat = MARKET_TO_STAT.get(market.get("key", ""))
            if not stat:
                continue

            # Group outcomes by player name → {Over: ..., Under: ...}
            by_player: dict[str, dict] = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                if not player:
                    continue
                direction = outcome.get("name", "")   # "Over" | "Under"
                by_player.setdefault(player, {})[direction] = {
                    "price": outcome.get("price"),
                    "point": outcome.get("point"),
                }

            for player, sides in by_player.items():
                over  = sides.get("Over", {})
                under = sides.get("Under", {})
                line  = over.get("point") or under.get("point")
                if line is None:
                    continue
                # Do not pair prices quoted at different thresholds.
                if over.get("point") is not None and under.get("point") is not None and over["point"] != under["point"]:
                    continue

                # Only write if we don't already have a preferred-book entry
                player_dict = out.setdefault(player, {})
                if stat not in player_dict:
                    player_dict[stat] = {
                        **metadata,
                        "updated_at": market.get("last_update") or book.get("last_update"),
                        "fetched_at": time.time(),
                        "line":        float(line),
                        "over_price":  over.get("price"),
                        "under_price": under.get("price"),
                        "bookmaker":   book_name,
                    }


def _get(url: str, params: dict) -> requests.Response | None:
    return odds_budget.request(requests.get, url, params)


def _track_quota(resp: requests.Response) -> None:
    """Record remaining API requests from response headers."""
    global _requests_remaining
    try:
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            _requests_remaining = int(remaining)
    except (ValueError, TypeError):
        pass


def _normalize_name(name: str) -> str:
    """
    Light normalization for name-matching edge cases.
    e.g. "P.J. Tucker" → "PJ Tucker", "De'Aaron Fox" stays as-is.
    """
    return name.replace(".", "").replace("  ", " ").strip()


# ── Utility: convert American odds ─────────────────────────────────────────

def american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds (e.g. -110, +150) to decimal odds.

    -110  →  1.909
    +150  →  2.500
    """
    if american_odds > 0:
        return round((american_odds / 100) + 1.0, 4)
    return round((100 / abs(american_odds)) + 1.0, 4)


def american_to_implied_prob(american_odds: int) -> float:
    """
    Convert American odds to implied probability (includes vig).

    -110  →  0.5238
    +150  →  0.4000
    """
    decimal = american_to_decimal(american_odds)
    return round(1 / decimal, 4)


def upcoming_events(events, target_date=None):
    """Only unstarted events on an exact Eastern calendar date; DST aware."""
    now = datetime.now(timezone.utc)
    target = target_date or now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    selected = []
    for event in events:
        try:
            start = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
            if event.get('id') and start.tzinfo and start > now and start.astimezone(ZoneInfo('America/New_York')).date().isoformat() == target:
                selected.append(event)
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(selected, key=lambda event: event['commence_time'])


def event_metadata(event, target_date, team_map):
    try:
        start = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        if start.tzinfo is None or not event.get('id'):
            return None
        event_date = start.astimezone(ZoneInfo('America/New_York')).date().isoformat()
        if target_date and event_date != target_date:
            return None
        home, away = team_map[event['home_team']], team_map[event['away_team']]
        return dict(event_id=event['id'], event_date=event_date, commence_time=event['commence_time'],
                    home_team=home, away_team=away, game_matchup=f'{away} @ {home}')
    except (KeyError, ValueError, TypeError):
        return None
