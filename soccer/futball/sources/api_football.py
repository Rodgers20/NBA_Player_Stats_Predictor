"""API-Football (api-sports.io) adapter — the keyed path.

Two things in this project genuinely require an API key, because no
public-domain source provides them:

* **UEFA Europa League** results and fixtures. openfootball publishes a
  Champions League repository but no Europa League one, and football-data.org's
  free tier does not include EL.
* **Live in-play events** for matches happening right now.

This adapter is inert without a key: `available()` returns False and the rest of
the system carries on with the key-free sources. Drop `API_FOOTBALL_KEY` into
`.env` and these endpoints light up with no other code change.

Free tier at time of writing: 100 requests/day, which is why every call here is
cached to the raw layer and re-read from disk rather than re-requested.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx

from .. import config
from ..storage import atomic_path, data_files
from . import _cache

log = logging.getLogger(__name__)

BASE = "https://v3.football.api-sports.io"
RAW_AF = config.RAW / "api-football"

# The free plan allows 10 requests/minute as well as 100/day. A 76-club squad
# backfill runs straight through that per-minute ceiling without a throttle, so
# requests are spaced rather than left to earn 429s.
MIN_REQUEST_INTERVAL = 6.5
_throttle_lock = threading.Lock()
_last_request = 0.0


def _throttle() -> None:
    global _last_request
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request = time.monotonic()

# The free tier allows 100 requests/day, so these TTLs are deliberately long.
# Under the two-source split, football-data.org carries the frequent results
# refresh and this key is reserved for goal and assist leaders, which change at
# most once a matchday.
FIXTURES_TTL_SECONDS = 3 * 3600
TOPSCORERS_TTL_SECONDS = 12 * 3600

# API-Football league ids for the competitions this project targets.
LEAGUE_IDS: dict[str, int] = {
    "EPL": 39,
    "SP1": 140,
    "D1": 78,
    "I1": 135,
    "F1": 61,
    "UCL": 2,
    "UEL": 3,     # Europa League — the one with no free alternative
    "MLS": 253,
}


def api_key() -> str | None:
    return os.environ.get("API_FOOTBALL_KEY") or None


def available() -> bool:
    return api_key() is not None


class MissingKey(RuntimeError):
    pass


def _get(path: str, params: dict[str, Any]) -> dict:
    key = api_key()
    if not key:
        raise MissingKey(
            "API_FOOTBALL_KEY is not set. Europa League data and live in-play "
            "events require a free key from https://www.api-football.com/ — "
            "add it to .env as API_FOOTBALL_KEY=..."
        )
    _throttle()
    with httpx.Client(timeout=45.0, headers={"x-apisports-key": key}) as client:
        resp = client.get(f"{BASE}{path}", params=params)
        if resp.status_code == 429:
            # Per-minute ceiling hit despite the throttle; wait it out once.
            log.warning("api-football rate limit hit, backing off")
            time.sleep(MIN_REQUEST_INTERVAL * 2)
            _throttle()
            resp = client.get(f"{BASE}{path}", params=params)
        resp.raise_for_status()
        body = resp.json()
    errors = body.get("errors")
    if errors:
        raise RuntimeError(f"api-football error: {errors}")
    return body


def status() -> dict:
    """Account status — the cheapest call, useful for verifying a key works."""
    return _get("/status", {}).get("response", {})


def _cache_is_fresh(dest, ttl: float) -> bool:
    """A cached payload may be reused only while it is younger than `ttl`.

    Without this the first fetch of a live season was kept forever: the file
    existed, so every later call short-circuited and the season never moved
    again. The 100-requests/day free tier makes caching necessary, but "cached"
    has to mean "recent", not "permanent".
    """
    if not dest.exists() or dest.stat().st_size == 0:
        return False
    if ttl <= 0:
        return False
    return (time.time() - dest.stat().st_mtime) < ttl


def fixtures(league_code: str, season: int, *, ttl: float = FIXTURES_TTL_SECONDS) -> list[dict]:
    """Season fixtures for one competition, cached to the raw layer with a TTL."""
    league_id = LEAGUE_IDS.get(league_code)
    if league_id is None:
        raise ValueError(f"no api-football league id mapped for {league_code}")

    dest = RAW_AF / "fixtures" / f"{league_code}-{season}.json"
    if _cache_is_fresh(dest, ttl):
        return json.loads(dest.read_text())

    try:
        body = _get("/fixtures", {"league": league_id, "season": season})
    except Exception:  # noqa: BLE001 - a stale season beats no season
        if dest.exists() and dest.stat().st_size > 0:
            log.warning("api-football fixtures %s failed; using cached payload", league_code)
            return json.loads(dest.read_text())
        raise

    rows = body.get("response", [])
    with atomic_path(dest) as temporary:
        temporary.write_text(json.dumps(rows))
    return rows


def live_fixtures() -> list[dict]:
    """Matches currently in play across all covered leagues."""
    return _get("/fixtures", {"live": "all"}).get("response", [])


def top_scorers(league_code: str, season: int) -> list[dict]:
    """Top scorers for a competition, normalised to this project's shape."""
    league_id = LEAGUE_IDS.get(league_code)
    if league_id is None:
        raise ValueError(f"no api-football league id mapped for {league_code}")
    dest = RAW_AF / "topscorers" / f"{league_code}-{season}.json"
    if _cache_is_fresh(dest, TOPSCORERS_TTL_SECONDS):
        rows = json.loads(dest.read_text())
    else:
        try:
            rows = _get("/players/topscorers",
                        {"league": league_id, "season": season}).get("response", [])
            with atomic_path(dest) as temporary:
                temporary.write_text(json.dumps(rows))
        except Exception:  # noqa: BLE001 - yesterday's leaders beat none
            if not (dest.exists() and dest.stat().st_size > 0):
                raise
            log.warning("api-football topscorers %s failed; using cached payload", league_code)
            rows = json.loads(dest.read_text())

    out = []
    for r in rows:
        player = r.get("player") or {}
        stats = (r.get("statistics") or [{}])[0]
        goals = stats.get("goals") or {}
        out.append({
            "player": player.get("name"),
            "photo": player.get("photo"),
            "team": (stats.get("team") or {}).get("name"),
            "goals": goals.get("total") or 0,
            "assists": goals.get("assists") or 0,
            "appearances": (stats.get("games") or {}).get("appearences") or 0,
        })
    return out


def to_result_rows(rows: list[dict], league_code: str, competition: str,
                   country: str, season_label: str) -> list[dict]:
    """Map api-football fixtures onto this project's result schema."""
    out = []
    for r in rows:
        fx = r.get("fixture", {})
        goals = r.get("goals", {})
        score = r.get("score", {}).get("halftime", {})
        teams = r.get("teams", {})
        if goals.get("home") is None or goals.get("away") is None:
            continue  # not played yet
        out.append({
            "competition_code": league_code,
            "competition": competition,
            "country": country,
            "season": season_label,
            "date": (fx.get("date") or "")[:10],
            "time": (fx.get("date") or "")[11:16] or None,
            "home_team": (teams.get("home") or {}).get("name"),
            "away_team": (teams.get("away") or {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "ht_home_goals": score.get("home"),
            "ht_away_goals": score.get("away"),
        })
    return out


# ---------------------------------------------------------------------------
# squads and player portraits
#
# The free plan is locked to seasons 2022-2024, which is why it cannot serve
# current results — but `/players/squads` is scoped by *team*, not by season,
# and is not gated. Verified 2026-09-13: it returns current squads with player
# ids and portrait URLs.
#
# Those portraits are the reason this exists. They are uniform 150x150 studio
# crops on a transparent background — the same framing for every player, in the
# current club kit. Wikipedia's portraits are larger but wildly inconsistent in
# crop, background and era, and it is the consistency that makes a row of
# players look professional rather than ragged.
#
# The media CDN itself needs no authentication, so downloading the images costs
# nothing against the 100/day request budget; only the squad lookups do.
# ---------------------------------------------------------------------------

# Squads change only in transfer windows, so this is deliberately long: at 76
# clubs a refresh would otherwise consume most of a day's request budget.
SQUADS_TTL_SECONDS = 7 * 24 * 3600
TEAM_IDS_TTL_SECONDS = 30 * 24 * 3600

# Season used purely to enumerate teams. Any season inside the free window
# works — team ids are stable, and the squad call that follows is season-free.
TEAM_LOOKUP_SEASON = 2024


def _teams_cache_path(league_code: str) -> Path:
    return RAW_AF / "teams" / f"{league_code}.json"


def fetch_team_ids(league_code: str, *, ttl: float = TEAM_IDS_TTL_SECONDS) -> dict[str, int]:
    """Club name -> API-Football team id for one competition.

    API-Football's team ids are its own; they do not match football-data.org's,
    so this mapping is what lets the squad lookups be driven from the club
    directory.
    """
    league_id = LEAGUE_IDS.get(league_code)
    if league_id is None:
        raise ValueError(f"no api-football league id mapped for {league_code}")

    dest = _teams_cache_path(league_code)
    if _cache.is_fresh(dest, ttl):
        cached = _cache.read_json(dest)
        if cached is not None:
            return {k: int(v) for k, v in cached.items()}

    try:
        body = _get("/teams", {"league": league_id, "season": TEAM_LOOKUP_SEASON})
    except Exception:  # noqa: BLE001 - a stale map beats none
        cached = _cache.read_json(dest)
        if cached is not None:
            log.warning("api-football teams %s failed; using cached map", league_code)
            return {k: int(v) for k, v in cached.items()}
        raise

    mapping = {
        (row.get("team") or {}).get("name"): (row.get("team") or {}).get("id")
        for row in body.get("response", [])
    }
    mapping = {k: int(v) for k, v in mapping.items() if k and v}
    _cache.write_json(dest, mapping)
    return mapping


def _squad_cache_path(team_id: int) -> Path:
    return RAW_AF / "squads" / f"{team_id}.json"


def squad_is_cached(team_id: int, *, ttl: float = SQUADS_TTL_SECONDS) -> bool:
    """True when this club's squad needs no request. Used to budget a backfill."""
    return _cache.is_fresh(_squad_cache_path(team_id), ttl)


def fetch_squad(team_id: int, *, ttl: float = SQUADS_TTL_SECONDS) -> list[dict]:
    """Current squad for one club: id, name, position and portrait URL."""
    dest = _squad_cache_path(team_id)
    if _cache.is_fresh(dest, ttl):
        cached = _cache.read_json(dest)
        if cached is not None:
            return cached

    try:
        body = _get("/players/squads", {"team": team_id})
    except Exception:  # noqa: BLE001 - a stale squad beats none
        cached = _cache.read_json(dest)
        if cached is not None:
            log.warning("api-football squad %s failed; using cached payload", team_id)
            return cached
        raise

    response = body.get("response") or []
    players = (response[0].get("players") or []) if response else []
    rows = [
        {
            "player_id": p.get("id"),
            "name": (p.get("name") or "").strip(),
            "age": p.get("age"),
            "number": p.get("number"),
            "position": p.get("position"),
            "photo": p.get("photo"),
            "team_id": team_id,
        }
        for p in players
        if (p.get("name") or "").strip()
    ]
    _cache.write_json(dest, rows)
    return rows


def read_squad(team_id: int) -> list[dict] | None:
    """Whatever squad is on disk, without touching the network."""
    return _cache.read_json(_squad_cache_path(team_id))


def cached_squads() -> list[dict]:
    """Every squad row currently on disk, across all clubs."""
    directory = RAW_AF / "squads"
    out: list[dict] = []
    if not directory.is_dir():
        return out
    for path in data_files(directory, "*.json"):
        rows = _cache.read_json(path)
        if isinstance(rows, list):
            out.extend(rows)
    return out


def requests_remaining() -> int | None:
    """How many requests today's plan has left, or None if it cannot be read.

    One cheap call, so a backfill can size itself instead of walking into the
    100/day wall halfway through.
    """
    try:
        body = status()
    except Exception:  # noqa: BLE001 - budgeting must not break the caller
        return None
    requests = body.get("requests") or {}
    limit, current = requests.get("limit_day"), requests.get("current")
    if not isinstance(limit, int) or not isinstance(current, int):
        return None
    return max(0, limit - current)


def search_team_id(name: str, *, ttl: float = TEAM_IDS_TTL_SECONDS) -> int | None:
    """Find a club's API-Football id by name.

    `fetch_team_ids` enumerates a league for one season, so it misses any club
    that was not in that league then — the 2026/27 promoted sides, and the
    Champions League entrants who play in competitions this project does not
    ingest. Searching by name reaches them at one request each.

    Results are cached individually, including misses, so a rerun costs nothing.
    """
    # The endpoint rejects anything but letters, digits and spaces, so accents
    # and punctuation are stripped first — "Málaga" and "Bodø/Glimt" both 400
    # otherwise.
    folded = unicodedata.normalize("NFKD", name)
    query = "".join(
        c for c in folded
        if (c.isalnum() and c.isascii()) or c.isspace()
    ).strip()
    query = " ".join(query.split())
    if len(query) < 3:
        return None

    slug = "".join(c if c.isalnum() else "-" for c in query.lower()).strip("-")
    dest = RAW_AF / "team-search" / f"{slug}.json"
    if _cache.is_fresh(dest, ttl):
        cached = _cache.read_json(dest)
        if cached is not None:
            return cached.get("team_id")

    try:
        body = _get("/teams", {"search": query})
    except Exception as exc:  # noqa: BLE001 - a miss must not stop a backfill
        log.warning("api-football team search %r failed: %s", name, exc)
        return None

    response = body.get("response") or []
    team_id = ((response[0].get("team") or {}).get("id")) if response else None
    _cache.write_json(dest, {"query": query, "team_id": team_id,
                             "matched": ((response[0].get("team") or {}).get("name")) if response else None})
    return team_id
