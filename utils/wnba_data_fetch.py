"""Live WNBA schedule and team roster helpers via nba_api (league_id=10)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_todays_wnba_games(target_date: Optional[str] = None) -> list[dict]:
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

    try:
        sb = scoreboardv3.ScoreboardV3(game_date=target_date, league_id="10")
        frames = sb.get_data_frames()
    except Exception as e:
        logger.warning(f"WNBA scoreboard fetch failed: {e}")
        return []

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
            "tip_time_et": str(row.get("gameEt", "")),
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
