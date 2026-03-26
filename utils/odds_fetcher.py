# utils/odds_fetcher.py
"""
Live Sportsbook Odds Fetcher
============================
Fetches real-time NBA player prop odds from The Odds API.

Setup:
  1. Sign up at https://the-odds-api.com  (free tier: 500 requests/month)
  2. Add to your .env file:
       THE_ODDS_API_KEY=your_key_here

The free tier is plenty for development — each full nightly refresh
costs ~10-15 API calls (one per game, all markets in one shot).
We cache results for 30 minutes so quota isn't burned on every page load.

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

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("THE_ODDS_API_KEY", "")
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
_cache_ts: float = 0.0     # unix timestamp of last fetch
_requests_remaining: int | None = None  # track quota from response headers

# Player props (per-event odds) require a paid plan and burn many API calls.
# Set to True to disable entirely — game odds (spread/totals/h2h) still work.
PLAYER_PROPS_ENABLED: bool = False

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

def get_live_odds(force_refresh: bool = False) -> dict:
    """
    Return live player prop odds for all today's NBA games.

    Uses a 30-minute in-memory cache.  Returns an empty dict when the API
    key is missing or the request fails — the rest of the pipeline degrades
    gracefully (falls back to estimated lines).

    Args:
        force_refresh: Bypass cache and re-fetch from API.

    Returns:
        Nested dict: {player_name: {stat: {"line", "over_price", "under_price", "bookmaker"}}}
    """
    global _cache, _cache_ts

    if not API_KEY:
        logger.debug("[OddsFetcher] THE_ODDS_API_KEY not set — skipping live odds")
        return {}

    if not force_refresh and _cache and (time.time() - _cache_ts) < _CACHE_TTL:
        return _cache

    global _player_props_unavailable

    if _player_props_unavailable:
        logger.debug("[OddsFetcher] Player props not available on current plan — skipping")
        return _cache

    logger.info("[OddsFetcher] Fetching fresh odds from The Odds API …")
    try:
        event_ids = _fetch_event_ids()
        if not event_ids:
            logger.info("[OddsFetcher] No upcoming NBA events found")
            return {}

        fresh: dict = {}
        markets = ",".join(MARKET_TO_STAT.keys())

        for event_id in event_ids:
            if _player_props_unavailable:
                break   # circuit breaker tripped mid-loop
            odds_data = _fetch_event_odds(event_id, markets)
            if odds_data:
                _parse_event_odds(odds_data, fresh)
            time.sleep(0.3)  # avoid 429 burst

        _cache = fresh
        _cache_ts = time.time()
        logger.info(f"[OddsFetcher] Cached odds for {len(fresh)} players "
                    f"({_requests_remaining} API requests remaining)")
        return _cache

    except Exception as exc:
        logger.warning(f"[OddsFetcher] Failed to fetch odds: {exc}")
        return _cache  # return stale cache rather than crashing


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

def _fetch_event_ids() -> list[str]:
    """Fetch today's NBA event IDs."""
    url = f"{BASE_URL}/sports/{SPORT}/events"
    resp = _get(url, {"apiKey": API_KEY, "dateFormat": "iso"})
    if resp is None:
        return []
    return [e["id"] for e in resp.json()]


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


def _parse_event_odds(event_data: dict, out: dict) -> None:
    """
    Parse one event's odds payload into the flat {player: {stat: odds}} dict.

    We iterate bookmakers in PREFERRED_BOOKS order so FanDuel lines win over
    less-popular books when both carry the same player prop.
    """
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

                # Only write if we don't already have a preferred-book entry
                player_dict = out.setdefault(player, {})
                if stat not in player_dict:
                    player_dict[stat] = {
                        "line":        float(line),
                        "over_price":  over.get("price", -110),
                        "under_price": under.get("price", -110),
                        "bookmaker":   book_name,
                    }


def _get(url: str, params: dict) -> requests.Response | None:
    """GET with error handling — returns None on failure."""
    global _player_props_unavailable
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 401:
            # Check whether this is a per-event player props call (paid plan required)
            if "/events/" in url and "/odds" in url:
                if not _player_props_unavailable:
                    logger.warning(
                        "[OddsFetcher] Player props (per-event) require a paid plan — "
                        "disabling for this session. Game odds (spreads/totals) still work."
                    )
                    _player_props_unavailable = True
            else:
                logger.error("[OddsFetcher] Invalid API key — check THE_ODDS_API_KEY in .env")
        elif status == 422:
            logger.debug("[OddsFetcher] No props available for event (422)")
        elif status == 429:
            logger.warning("[OddsFetcher] Rate limited (429) — too many requests in a short window")
        else:
            logger.warning(f"[OddsFetcher] HTTP error {e}")
        return None
    except requests.RequestException as e:
        logger.warning(f"[OddsFetcher] Request failed: {e}")
        return None


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
