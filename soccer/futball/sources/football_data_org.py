"""football-data.org (v4) ingestion — the gap-filling results source.

Why this exists
---------------
football-data.co.uk regenerates its season CSV only after a round completes, so
between a match finishing and that file being rebuilt — usually a day or two —
the league table is silently a matchday behind. openfootball lags identically,
and its `champions-league` repository stops at 2025-26 altogether, which is why
the Champions League had no current season at all.

football-data.org publishes both, on a free tier that covers exactly the three
competitions this dashboard shows. Verified against the live `/v4/competitions`
list on 2026-09-13: the free plan (`TIER_ONE`) carries twelve competitions
including `PL`, `PD` and `CL`.

What this module is *not*
-------------------------
It is not the canonical source. football-data.co.uk stays canonical because it
carries shots, cards, xG and closing odds, and because every team name already
in the lake is spelled its way. This module supplies goals and nothing else, and
`transform.results` reconciles its team names before merging so the same match
from both sources collapses to one row rather than double-counting.

Rate limits
-----------
Free tier is 10 requests/minute with no published daily cap, so the throttle
here is per-minute rather than per-day. Three competitions per refresh means a
refresh costs three requests; at the default 15-minute TTL that is 288/day, far
inside the ceiling.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from .. import config
from . import _cache

log = logging.getLogger(__name__)

BASE = "https://api.football-data.org/v4"
RAW_FDORG = config.RAW / "football-data-org"

# This project's competition code -> football-data.org competition code.
# All three are on the free tier; see the module docstring.
COMPETITIONS: dict[str, str] = {
    "E0": "PL",    # Premier League
    "SP1": "PD",   # Primera Division (La Liga)
    "D1": "BL1",   # Bundesliga (entitled on this token; verified 2026-11-03)
    "UCL": "CL",   # UEFA Champions League
}  # MLS is not offered by football-data.org at all (403), so it comes from ESPN.

# Statuses that mean "this match has a final score".
PLAYED_STATUSES = frozenset({"FINISHED", "AWARDED"})

# 15 minutes: fresh enough that a full-time score appears within a quarter of an
# hour, cheap enough to be a rounding error against the rate limit.
CACHE_TTL_SECONDS = 900

# 10 requests/minute => one every 6 seconds. A little margin on top.
MIN_REQUEST_INTERVAL = 6.5

_throttle_lock = threading.Lock()
_last_request = 0.0


class MissingToken(RuntimeError):
    """No football-data.org token configured."""


class NotEntitled(RuntimeError):
    """The token is valid but this competition is outside its plan."""


def api_token() -> str | None:
    return os.environ.get("FOOTBALL_DATA_ORG_KEY") or None


def available() -> bool:
    return api_token() is not None


# ---------------------------------------------------------------------------
# season labels
#
# The lake already contains two spellings, and they are not interchangeable:
# football-data.co.uk divisions are written "2026/27" by results._season_label,
# while Champions League seasons come from openfootball *filenames* and are
# written "2026-27". Emitting the wrong one splits a season in two.
# ---------------------------------------------------------------------------

def season_label(competition_code: str, start_year: int) -> str:
    end = f"{(start_year + 1) % 100:02d}"
    separator = "-" if competition_code == "UCL" else "/"
    return f"{start_year}{separator}{end}"


def _season_start_year(match: dict, fallback_date: str) -> int:
    start = ((match.get("season") or {}).get("startDate") or "")[:4]
    if start.isdigit():
        return int(start)
    # No season block: infer from the kick-off, European seasons starting in July.
    year, month = int(fallback_date[:4]), int(fallback_date[5:7])
    return year if month >= 7 else year - 1


def _int(value: Any) -> int | None:
    # bool is a subclass of int, so `isinstance(True, int)` is True and a JSON
    # `true` would otherwise arrive as a goal count of 1.
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def parse_matches(payload: dict, competition_code: str) -> list[dict]:
    """Map a `/matches` payload onto plain result rows.

    Only matches with a final score contribute. An unplayed, postponed or
    in-progress match yields nothing — a partial score written into the results
    lake would be indistinguishable from a full-time one.
    """
    rows: list[dict] = []
    for match in (payload or {}).get("matches") or []:
        if match.get("status") not in PLAYED_STATUSES:
            continue

        score = match.get("score") or {}
        full = score.get("fullTime") or {}
        home_goals, away_goals = _int(full.get("home")), _int(full.get("away"))
        if home_goals is None or away_goals is None:
            continue

        utc = match.get("utcDate") or ""
        # match_key embeds this date, so it must agree with football-data.co.uk's
        # local match date or the same fixture yields two keys. Every competition
        # mapped here is European (UTC to UTC+2), and football kicks off between
        # roughly 12:00 and 22:00 local, so the local day and the UTC day are
        # always the same one. Adding a competition outside that band would break
        # the assumption — see test_utc_date_matches_local_date_for_late_kickoffs.
        date = utc[:10]
        if len(date) != 10:
            continue

        home = ((match.get("homeTeam") or {}).get("name") or "").strip()
        away = ((match.get("awayTeam") or {}).get("name") or "").strip()
        if not home or not away:
            continue

        half = score.get("halfTime") or {}
        rows.append({
            "date": date,
            "time": utc[11:16] or None,
            "home_team": home,
            "away_team": away,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "ht_home_goals": _int(half.get("home")),
            "ht_away_goals": _int(half.get("away")),
            "season": season_label(competition_code, _season_start_year(match, date)),
            "matchday": match.get("matchday"),
            "stage": match.get("stage"),
        })
    return rows


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _throttle() -> None:
    global _last_request
    with _throttle_lock:
        elapsed = time.monotonic() - _last_request
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request = time.monotonic()


def _get(path: str, params: dict[str, Any]) -> dict:
    token = api_token()
    if not token:
        raise MissingToken(
            "FOOTBALL_DATA_ORG_KEY is not set. Get a free token at "
            "https://www.football-data.org/client/register and add it to .env "
            "as FOOTBALL_DATA_ORG_KEY=..."
        )
    _throttle()
    with httpx.Client(timeout=45.0, headers={"X-Auth-Token": token}) as client:
        response = client.get(f"{BASE}{path}", params=params)
        if response.status_code == 429:
            # The API tells us exactly how long the counter has left.
            wait = float(response.headers.get("X-RequestCounter-Reset", 60))
            log.warning("football-data.org rate limit hit, waiting %.0fs", wait)
            time.sleep(min(wait, 60) + 1)
            _throttle()
            response = client.get(f"{BASE}{path}", params=params)
        if response.status_code == 403:
            raise NotEntitled(
                f"football-data.org refused {path} (403). The free tier covers "
                f"only the twelve TIER_ONE competitions."
            )
        response.raise_for_status()
        return response.json()


def _cache_path(competition_code: str) -> Path:
    return RAW_FDORG / "matches" / f"{competition_code}.json"


def fetch_matches(
    competition_code: str,
    season: int | None = None,
    *,
    ttl: float = CACHE_TTL_SECONDS,
) -> dict:
    """Season matches for one competition, cached to the raw layer with a TTL.

    Unlike a permanent cache, this expires — a live season must be allowed to
    move. When a refetch fails but a cached payload exists, the stale payload is
    returned: a table a few hours old beats no table at all.
    """
    remote = COMPETITIONS.get(competition_code)
    if remote is None:
        raise ValueError(
            f"{competition_code} is not mapped to a football-data.org competition; "
            f"known: {sorted(COMPETITIONS)}"
        )

    dest = _cache_path(competition_code)
    cached = _cache.read_json(dest)
    if cached is not None and _cache.is_fresh(dest, ttl):
        return cached

    params: dict[str, Any] = {}
    if season is not None:
        params["season"] = season

    try:
        payload = _get(f"/competitions/{remote}/matches", params)
    except (MissingToken, NotEntitled):
        # A missing token or a competition outside the plan is a configuration
        # error, not a transient failure. Returning a stale payload here would
        # hide it behind data that silently stops updating.
        raise
    except Exception as exc:  # noqa: BLE001 - any transport failure is transient
        if cached is not None:
            log.warning("football-data.org %s failed (%s); using cached payload",
                        competition_code, exc, exc_info=True)
            return cached
        raise

    _cache.write_json(dest, payload)
    return payload


def read_matches(competition_code: str) -> dict | None:
    """Whatever is on disk for this competition, without touching the network."""
    return _cache.read_json(_cache_path(competition_code))


def ingest(
    codes: list[str] | None = None,
    *,
    season: int | None = None,
    ttl: float = CACHE_TTL_SECONDS,
) -> dict[str, int]:
    """Refresh every mapped competition. Inert without a token."""
    report = {"fetched": 0, "matches": 0, "skipped": 0, "failed": 0}
    if not available():
        log.info("football-data.org: no FOOTBALL_DATA_ORG_KEY, skipping")
        report["skipped"] = len(codes or COMPETITIONS)
        return report

    config.ensure_dirs()
    for code in (codes or list(COMPETITIONS)):
        try:
            payload = fetch_matches(code, season=season, ttl=ttl)
        except Exception as exc:  # noqa: BLE001 - one bad competition must not stop the rest
            log.warning("football-data.org %s failed: %s", code, exc, exc_info=True)
            report["failed"] += 1
            continue
        report["fetched"] += 1
        report["matches"] += len(parse_matches(payload, code))
    return report


# ---------------------------------------------------------------------------
# scorers
#
# The free tier does serve this endpoint for the current season, despite the
# pricing page listing goal scorers under the paid Deep Data plan. Verified
# 2026-09-13 against PL and PD.
#
# One real limitation: `assists` is null for most players — 8 of 30 populated
# on La Liga when checked. That is upstream coverage, not a parsing failure, so
# a null is carried through as None rather than coerced to 0. "No assists
# recorded" and "zero assists" are different facts and the UI must be able to
# tell them apart.
# ---------------------------------------------------------------------------

SCORERS_TTL_SECONDS = 900
SCORERS_LIMIT = 100  # the endpoint caps out around 90 rows; 50 was
# silently dropping real contributors — Pedri and Xavi Espart are both
# Barcelona league scorers who fell outside the top 50.


def _scorers_cache_path(competition_code: str) -> Path:
    return RAW_FDORG / "scorers" / f"{competition_code}.json"


def fetch_scorers(
    competition_code: str,
    *,
    limit: int = SCORERS_LIMIT,
    ttl: float = SCORERS_TTL_SECONDS,
) -> dict:
    """Top scorers for one competition, cached with a TTL like `fetch_matches`."""
    remote = COMPETITIONS.get(competition_code)
    if remote is None:
        raise ValueError(
            f"{competition_code} is not mapped to a football-data.org competition; "
            f"known: {sorted(COMPETITIONS)}"
        )

    dest = _scorers_cache_path(competition_code)
    cached = _cache.read_json(dest)
    if cached is not None and _cache.is_fresh(dest, ttl):
        return cached

    try:
        payload = _get(f"/competitions/{remote}/scorers", {"limit": limit})
    except (MissingToken, NotEntitled):
        raise
    except Exception as exc:  # noqa: BLE001 - transport failures are transient
        if cached is not None:
            log.warning("football-data.org scorers %s failed (%s); using cache",
                        competition_code, exc, exc_info=True)
            return cached
        raise

    _cache.write_json(dest, payload)
    return payload


def read_scorers(competition_code: str) -> dict | None:
    """Cached scorers payload, without touching the network."""
    return _cache.read_json(_scorers_cache_path(competition_code))


def parse_scorers(payload: dict, competition_code: str) -> list[dict]:
    """Normalise a `/scorers` payload into leaderboard rows."""
    rows: list[dict] = []
    for entry in (payload or {}).get("scorers") or []:
        player = entry.get("player") or {}
        team = entry.get("team") or {}
        name = (player.get("name") or "").strip()
        if not name:
            continue
        rows.append({
            "player_id": player.get("id"),
            "player": name,
            "team": (team.get("shortName") or team.get("name") or "").strip() or None,
            "team_id": team.get("id"),
            "crest": team.get("crest"),
            "nationality": player.get("nationality"),
            "date_of_birth": player.get("dateOfBirth"),
            "section": player.get("section"),
            "shirt_number": player.get("shirtNumber"),
            "goals": _int(entry.get("goals")) or 0,
            # None means "not published", which is not the same as zero.
            "assists": _int(entry.get("assists")),
            "penalties": _int(entry.get("penalties")),
            "played": _int(entry.get("playedMatches")),
            "competition_code": competition_code,
        })
    return rows


def ingest_scorers(codes: list[str] | None = None, *, ttl: float = SCORERS_TTL_SECONDS) -> dict[str, int]:
    """Refresh the scorer boards. Inert without a token."""
    report = {"fetched": 0, "players": 0, "skipped": 0, "failed": 0}
    if not available():
        report["skipped"] = len(codes or COMPETITIONS)
        return report
    config.ensure_dirs()
    for code in (codes or list(COMPETITIONS)):
        try:
            payload = fetch_scorers(code, ttl=ttl)
        except Exception as exc:  # noqa: BLE001 - one board must not stop the rest
            log.warning("football-data.org scorers %s failed: %s", code, exc, exc_info=True)
            report["failed"] += 1
            continue
        report["fetched"] += 1
        report["players"] += len(parse_scorers(payload, code))
    return report


# ---------------------------------------------------------------------------
# squads and crests
#
# The dashboard previously fell back to drawing a club's initials whenever a
# badge could not be resolved, which produced "BA" for Barça and "C" for Celta.
# This endpoint returns every club in a competition with an official crest URL,
# so the fallback becomes the exception rather than the norm.
#
# Squads are stable within a season, so the TTL is long.
# ---------------------------------------------------------------------------

TEAMS_TTL_SECONDS = 24 * 3600


def _teams_cache_path(competition_code: str) -> Path:
    return RAW_FDORG / "teams" / f"{competition_code}.json"


def fetch_teams(competition_code: str, *, ttl: float = TEAMS_TTL_SECONDS) -> dict:
    """Clubs in one competition, with crest URLs. Cached with a TTL."""
    remote = COMPETITIONS.get(competition_code)
    if remote is None:
        raise ValueError(f"{competition_code} is not mapped; known: {sorted(COMPETITIONS)}")

    dest = _teams_cache_path(competition_code)
    cached = _cache.read_json(dest)
    if cached is not None and _cache.is_fresh(dest, ttl):
        return cached

    try:
        payload = _get(f"/competitions/{remote}/teams", {})
    except (MissingToken, NotEntitled):
        raise
    except Exception as exc:  # noqa: BLE001 - transport failures are transient
        if cached is not None:
            log.warning("football-data.org teams %s failed (%s); using cache",
                        competition_code, exc, exc_info=True)
            return cached
        raise

    _cache.write_json(dest, payload)
    return payload


def read_teams(competition_code: str) -> dict | None:
    return _cache.read_json(_teams_cache_path(competition_code))


def parse_teams(payload: dict) -> list[dict]:
    """Normalise a `/teams` payload into crest-bearing rows."""
    rows: list[dict] = []
    for team in (payload or {}).get("teams") or []:
        name = (team.get("name") or "").strip()
        if not name:
            continue
        coach = team.get("coach") or {}
        area = team.get("area") or {}
        rows.append({
            "team_id": team.get("id"),
            "name": name,
            "short_name": (team.get("shortName") or "").strip() or None,
            "tla": (team.get("tla") or "").strip() or None,
            "crest": team.get("crest"),
            "founded": team.get("founded"),
            "venue": team.get("venue"),
            "website": team.get("website"),
            "address": team.get("address"),
            "club_colors": team.get("clubColors"),
            "country": (area.get("name") or "").strip() or None,
            "country_flag": area.get("flag"),
            "coach": {
                "name": (coach.get("name") or "").strip() or None,
                "nationality": coach.get("nationality"),
                "date_of_birth": coach.get("dateOfBirth"),
                "contract_until": (coach.get("contract") or {}).get("until"),
            } if coach.get("name") else None,
            "squad": [
                {
                    "player_id": p.get("id"),
                    "name": (p.get("name") or "").strip(),
                    "position": p.get("position"),
                    "date_of_birth": p.get("dateOfBirth"),
                    "nationality": p.get("nationality"),
                }
                for p in (team.get("squad") or [])
                if (p.get("name") or "").strip()
            ],
        })
    return rows


def ingest_teams(codes: list[str] | None = None, *, ttl: float = TEAMS_TTL_SECONDS) -> dict[str, int]:
    """Refresh club directories. Inert without a token."""
    report = {"fetched": 0, "teams": 0, "players": 0, "skipped": 0, "failed": 0}
    if not available():
        report["skipped"] = len(codes or COMPETITIONS)
        return report
    config.ensure_dirs()
    for code in (codes or list(COMPETITIONS)):
        try:
            payload = fetch_teams(code, ttl=ttl)
        except Exception as exc:  # noqa: BLE001 - one competition must not stop the rest
            log.warning("football-data.org teams %s failed: %s", code, exc, exc_info=True)
            report["failed"] += 1
            continue
        rows = parse_teams(payload)
        report["fetched"] += 1
        report["teams"] += len(rows)
        report["players"] += sum(len(r["squad"]) for r in rows)
    return report


# Matches that have not been played. `parse_matches` deliberately drops these —
# a partial or absent score must never enter the results lake — but they are
# exactly what the fixture layer needs, and for the Champions League they are
# the only source: football-data.co.uk's rolling file and openfootball's season
# schedules both cover domestic divisions only.
SCHEDULED_STATUSES = frozenset({"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"})


def parse_fixtures(payload: dict, competition_code: str) -> list[dict]:
    """Upcoming matches from a `/matches` payload, shaped for the fixture layer."""
    rows: list[dict] = []
    for match in (payload or {}).get("matches") or []:
        if match.get("status") not in SCHEDULED_STATUSES:
            continue

        utc = match.get("utcDate") or ""
        date = utc[:10]
        if len(date) != 10:
            continue

        home = ((match.get("homeTeam") or {}).get("name") or "").strip()
        away = ((match.get("awayTeam") or {}).get("name") or "").strip()
        if not home or not away:
            continue

        matchday = match.get("matchday")
        rows.append({
            "competition_code": competition_code,
            "date": date,
            "time": utc[11:16] or None,
            "home_team": home,
            "away_team": away,
            "round": f"Matchday {matchday}" if matchday else match.get("stage"),
            "home_goals": None,
            "away_goals": None,
            "source": "football-data.org",
        })
    return rows


def read_fixtures(competition_code: str) -> list[dict]:
    """Upcoming matches from the cached payload, without touching the network."""
    payload = read_matches(competition_code)
    return parse_fixtures(payload, competition_code) if payload else []


# --------------------------------------------------------------------------
# live scores
#
# The results cache has a 15-minute TTL, which is right for a league table and
# far too slow for a match in progress. Live scores therefore get their own
# cache file and a much shorter TTL. That is affordable precisely here:
# football-data.org allows 10 requests a minute with **no daily cap**, so three
# competitions polled every 45 seconds costs 4 requests a minute at worst —
# whereas API-Football's 100-a-day ceiling makes live polling impossible.
# --------------------------------------------------------------------------

#: A live score is worthless stale, and the quota allows this.
LIVE_TTL_SECONDS = 45

#: Statuses where a score exists and is still moving.
IN_PLAY_STATUSES = frozenset({"IN_PLAY", "PAUSED", "LIVE"})


def _live_cache_path(competition_code: str) -> Path:
    return RAW_FDORG / "live" / f"{competition_code}.json"


def fetch_live(competition_code: str, *, ttl: float = LIVE_TTL_SECONDS) -> dict:
    """In-play matches for one competition, cached with a short TTL."""
    remote = COMPETITIONS.get(competition_code)
    if remote is None:
        raise ValueError(f"{competition_code} is not mapped; known: {sorted(COMPETITIONS)}")

    dest = _live_cache_path(competition_code)
    cached = _cache.read_json(dest)
    if cached is not None and _cache.is_fresh(dest, ttl):
        return cached

    try:
        payload = _get(f"/competitions/{remote}/matches", {"status": "IN_PLAY"})
    except (MissingToken, NotEntitled):
        raise
    except Exception as exc:  # noqa: BLE001 - transport failures are transient
        if cached is not None:
            log.warning("football-data.org live %s failed (%s); using cache",
                        competition_code, exc, exc_info=True)
            return cached
        raise

    _cache.write_json(dest, payload)
    return payload


def read_live(competition_code: str) -> dict | None:
    """Cached live payload, without touching the network."""
    return _cache.read_json(_live_cache_path(competition_code))


def parse_live(payload: dict, competition_code: str) -> list[dict]:
    """Running scores for matches actually in play.

    A match is only reported when the provider says it is in play **and** both
    goal figures are present. A null score on an in-play match means the feed
    has not caught up, and publishing it as 0-0 would invent a scoreline.
    """
    rows = []
    for match in (payload or {}).get("matches") or []:
        if match.get("status") not in IN_PLAY_STATUSES:
            continue
        score = ((match.get("score") or {}).get("fullTime")) or {}
        home, away = score.get("home"), score.get("away")
        if home is None or away is None:
            continue
        rows.append({
            "competition_code": competition_code,
            "provider_id": match.get("id"),
            "utc_date": match.get("utcDate"),
            "status": match.get("status"),
            "home_name": (match.get("homeTeam") or {}).get("name"),
            "away_name": (match.get("awayTeam") or {}).get("name"),
            "home_goals": int(home),
            "away_goals": int(away),
        })
    return rows
