"""Live WNBA schedule and team roster helpers via nba_api (league_id=10)."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Schedule cache ───────────────────────────────────────────────────────────
# The scoreboard endpoint costs ~0.6s per call and dashboard/app.py invokes it
# from six separate places per WNBA render (plus once inside
# get_matchup_for_date), which is why the WNBA page lagged the NBA one. A short
# TTL keeps live scores fresh while collapsing a render into a single request.
_SCHEDULE_TTL = 120  # seconds
_schedule_cache: dict[str, tuple[float, list[dict]]] = {}


def clear_schedule_cache() -> None:
    """Drop all cached schedule entries (used by tests and manual refresh)."""
    _schedule_cache.clear()


def get_todays_wnba_games(
    target_date: Optional[str] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Cached wrapper around _fetch_todays_wnba_games.

    A successful fetch is cached for _SCHEDULE_TTL seconds, including one that
    legitimately returns zero games (offseason / no slate) — that is a real
    answer and re-querying it every render is what made the page slow. A failed
    fetch is NOT cached, so transient network errors are retried rather than
    pinned as an empty schedule for the whole TTL.
    """
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    now = time.monotonic()
    if not force_refresh:
        hit = _schedule_cache.get(target_date)
        if hit is not None and (now - hit[0]) < _SCHEDULE_TTL:
            return hit[1]

    games = _fetch_todays_wnba_games(target_date)
    if games is None:          # fetch failed — do not poison the cache
        return []

    _schedule_cache[target_date] = (now, games)
    return games


_STATS_TIMEOUT = 8      # seconds to wait on stats.nba.com before giving up
_STATS_BACKOFF = 600    # seconds to skip it after a failure
_stats_down_until = 0.0


def _espn_wnba_games(target_date: str) -> Optional[list[dict]]:
    """Same game dicts from ESPN's scoreboard; None if that fails too."""
    from utils.basketball_content import _fetch
    from utils.espn_game_odds import _team_abbr
    try:
        board = _fetch("wnba", "scoreboard", dates=target_date.replace("-", ""))
    except Exception as e:
        logger.warning(f"WNBA ESPN scoreboard fetch failed: {e}")
        return None
    games = []
    for event in board.get("events") or []:
        comp = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
        if "home" not in sides or "away" not in sides:
            continue
        status = (comp.get("status") or event.get("status") or {}).get("type") or {}
        state = status.get("state", "pre")

        def side(competitor):
            team = competitor.get("team") or {}
            overall = next((r.get("summary") for r in competitor.get("records") or []
                            if r.get("type") == "total"), "") or ""
            wins, _, losses = overall.partition("-")
            return {"team_id": int(team["id"]) if str(team.get("id", "")).isdigit() else None,
                    "abbrev": _team_abbr(competitor, "wnba"), "name": team.get("displayName", ""),
                    "wins": int(wins) if wins.isdigit() else 0, "losses": int(losses) if losses.isdigit() else 0,
                    "score": int(competitor["score"]) if state != "pre" and str(competitor.get("score", "")).isdigit() else None}

        games.append({"game_id": str(event.get("id")),
                      "status": {"pre": "Scheduled", "in": "In Progress", "post": "Final"}.get(state, "Scheduled"),
                      "status_text": status.get("shortDetail") or status.get("description") or "",
                      "tip_time_et": event.get("date", ""), "home": side(sides["home"]), "away": side(sides["away"])})
    return games


def _et_iso(value) -> str:
    """The scoreboard's gameEt is Eastern wall-clock time with a bogus trailing Z; return a real offset."""
    try:
        naive = datetime.fromisoformat(str(value).replace("Z", ""))
    except ValueError:
        return str(value or "")
    return naive.replace(tzinfo=ZoneInfo("America/New_York")).isoformat()


def _fetch_todays_wnba_games(target_date: Optional[str] = None) -> Optional[list[dict]]:
    """Return today's WNBA games (or games for target_date="YYYY-MM-DD").

    Each game dict shape:
        {
          "game_id": str,
          "status": str,        # "Scheduled" / "In Progress" / "Final"
          "status_text": str,   # "7:00 pm ET" or "Q3 5:24" or "Final"
          "tip_time_et": str,   # ISO 8601 ET
          "home": {"team_id": int, "abbrev": str, "name": str, "wins": int, "losses": int, "score": int|None},
          "away": {...},
        }
    """
    from nba_api.stats.endpoints import scoreboardv3

    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    global _stats_down_until
    if time.monotonic() < _stats_down_until:
        return _espn_wnba_games(target_date)
    try:
        sb = scoreboardv3.ScoreboardV3(game_date=target_date, league_id="10", timeout=_STATS_TIMEOUT)
        frames = sb.get_data_frames()
    except Exception as e:
        # stats.nba.com blocks cloud runners; stop waiting on it for a while and use ESPN.
        logger.warning(f"WNBA scoreboard fetch failed: {e}")
        _stats_down_until = time.monotonic() + _STATS_BACKOFF
        return _espn_wnba_games(target_date)

    if len(frames) < 3:
        return []

    games_df = frames[1]           # GameHeader-ish
    team_lines_df = frames[2]      # LineScore (both teams per game)

    if games_df.empty:
        return []

    # Build team-side lookup: {game_id: {"home": {...}, "away": {...}}}
    sides: dict[str, dict[str, dict]] = {}
    for game_id, group in team_lines_df.groupby("gameId"):
        game_meta = games_df[games_df["gameId"] == game_id]
        if game_meta.empty:
            continue
        home_team_id = int(game_meta.iloc[0]["homeTeamId"]) if "homeTeamId" in game_meta.columns else None
        home_side, away_side = None, None
        for _, row in group.iterrows():
            side = {
                "team_id": int(row["teamId"]),
                "abbrev": str(row.get("teamTricode", "")),
                "name": f"{row.get('teamCity','')} {row.get('teamName','')}".strip(),
                "wins": int(row.get("wins", 0) or 0),
                "losses": int(row.get("losses", 0) or 0),
                "score": None if pd.isna(row.get("score")) or row.get("score") == 0 else int(row["score"]),
            }
            if home_team_id is not None and int(row["teamId"]) == home_team_id:
                home_side = side
            else:
                away_side = side
        # Fallback if homeTeamId column missing: first row = home by convention
        rows = list(group.itertuples())
        if home_side is None and rows:
            home_side = {
                "team_id": int(rows[0].teamId), "abbrev": str(getattr(rows[0], "teamTricode", "")),
                "name": f"{getattr(rows[0], 'teamCity', '')} {getattr(rows[0], 'teamName', '')}".strip(),
                "wins": int(getattr(rows[0], "wins", 0) or 0),
                "losses": int(getattr(rows[0], "losses", 0) or 0),
                "score": None,
            }
        if away_side is None and len(rows) > 1:
            away_side = {
                "team_id": int(rows[1].teamId), "abbrev": str(getattr(rows[1], "teamTricode", "")),
                "name": f"{getattr(rows[1], 'teamCity', '')} {getattr(rows[1], 'teamName', '')}".strip(),
                "wins": int(getattr(rows[1], "wins", 0) or 0),
                "losses": int(getattr(rows[1], "losses", 0) or 0),
                "score": None,
            }
        sides[str(game_id)] = {"home": home_side, "away": away_side}

    games: list[dict] = []
    for _, row in games_df.iterrows():
        gid = str(row["gameId"])
        side_pair = sides.get(gid, {"home": None, "away": None})
        if not side_pair["home"] or not side_pair["away"]:
            continue
        status_id = int(row.get("gameStatus", 1))
        status = {1: "Scheduled", 2: "In Progress", 3: "Final"}.get(status_id, "Unknown")
        games.append({
            "game_id": gid,
            "status": status,
            "status_text": str(row.get("gameStatusText", "")).strip(),
            "tip_time_et": _et_iso(row.get("gameEt", "")),
            "home": side_pair["home"],
            "away": side_pair["away"],
        })

    return games


def get_teams_playing_today_wnba(target_date: Optional[str] = None) -> set[str]:
    """Return set of team tricodes playing today (for filtering rosters)."""
    games = get_todays_wnba_games(target_date)
    playing = set()
    for g in games:
        if g["home"] and g["home"]["abbrev"]:
            playing.add(g["home"]["abbrev"])
        if g["away"] and g["away"]["abbrev"]:
            playing.add(g["away"]["abbrev"])
    return playing
