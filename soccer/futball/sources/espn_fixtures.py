"""Fixtures for competitions football-data.org does not serve (MLS), read from ESPN's scoreboard.

One single-date request per day, because ESPN answers 400 to a multi-day range. Boxscore's
cached `fetch_scoreboard` is reused so the same day is never downloaded twice.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from .. import boxscores

log = logging.getLogger(__name__)

#: Competitions whose fixtures come only from ESPN.
ESPN_ONLY: tuple[str, ...] = ("MLS",)


def parse(board: dict, code: str) -> list[dict]:
    """Every event on a scoreboard as a fixture record (score only once it is final)."""
    out: list[dict] = []
    for event in (board or {}).get("events") or []:
        competition = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in competition.get("competitors") or []}
        if "home" not in sides or "away" not in sides:
            continue
        final = bool(((event.get("status") or {}).get("type") or {}).get("completed"))
        stamp = str(event.get("date", ""))
        out.append({
            "competition_code": code, "date": stamp[:10], "time": stamp[11:16] or None,
            "home_team": sides["home"].get("team", {}).get("displayName", ""),
            "away_team": sides["away"].get("team", {}).get("displayName", ""),
            "home_goals": boxscores._int(sides["home"].get("score")) if final else None,
            "away_goals": boxscores._int(sides["away"].get("score")) if final else None,
            "source": "ESPN", "referee": None, "round": None,
            "odds_home": None, "odds_draw": None, "odds_away": None,
        })
    return out


def read(code: str, today: date | None = None, days: int = 21) -> list[dict]:
    """Fixture records from today through `days` ahead. A failed day is logged and skipped."""
    today = today or datetime.now(timezone.utc).date()
    out: list[dict] = []
    for offset in range(days + 1):
        day = today + timedelta(days=offset)
        try:
            out.extend(parse(boxscores.fetch_scoreboard(code, day, today), code))
        except RuntimeError as exc:
            log.warning("ESPN fixtures %s %s unavailable: %s", code, day, exc)
    return out
