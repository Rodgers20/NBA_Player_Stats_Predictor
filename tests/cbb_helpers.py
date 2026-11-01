"""Small offline fixtures for the cbbpred tests: ESPN-shaped events and a synthetic season."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from cbbpred import espn

UTC = timezone.utc
CONFS = {"2": {"name": "Atlantic Coast Conference", "short": "ACC", "logo": None},
         "7": {"name": "Big Ten Conference", "short": "Big Ten", "logo": None}}


def event(eid: str, when: str, home: tuple[str, str, str | None], away: tuple[str, str, str | None],
          hs: int = 0, as_: int = 0, state: str = "post", name: str = "STATUS_FINAL", neutral: bool = False,
          conf_game: bool = False, rank: int = 99, period: int = 2) -> dict[str, Any]:
    """An ESPN scoreboard event. Teams are (id, abbr, conference id)."""
    def side(kind: str, t: tuple[str, str, str | None], score: int) -> dict[str, Any]:
        team = {"id": t[0], "abbreviation": t[1], "displayName": f"{t[1]} Team", "shortDisplayName": t[1]}
        if t[2]:
            team["conferenceId"] = t[2]
        return {"homeAway": kind, "team": team, "score": str(score), "curatedRank": {"current": rank}}
    return {"id": eid, "date": when, "season": {"year": 2027, "type": 2},
            "competitions": [{"neutralSite": neutral, "conferenceCompetition": conf_game, "notes": [],
                              "venue": {"fullName": "Arena"}, "competitors": [side("home", home, hs), side("away", away, as_)],
                              "status": {"period": period, "type": {"name": name, "state": state,
                                                                    "completed": state == "post" and name == "STATUS_FINAL"}}}]}


A, B, C = ("1", "AAA", "2"), ("2", "BBB", "2"), ("3", "CCC", "7")


def games_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=espn.GAME_SCHEMA)


def season_games(n_days: int = 12, start: datetime = datetime(2026, 11, 3, 23, 0, tzinfo=UTC)) -> pl.DataFrame:
    """A rotating A/B/C schedule: A usually beats B, B usually beats C; the last day is unplayed."""
    pairs = [(A, B, 80, 70), (B, C, 75, 65), (C, A, 60, 78)]
    rows = []
    for d in range(n_days):
        home, away, hs, as_ = pairs[d % 3]
        when = (start + timedelta(days=d)).strftime("%Y-%m-%dT%H:%MZ")
        last = d == n_days - 1
        ev = event(str(1000 + d), when, home, away, 0 if last else hs, 0 if last else as_,
                   state="pre" if last else "post", name="STATUS_SCHEDULED" if last else "STATUS_FINAL",
                   conf_game=home[2] == away[2])
        rows.append(espn.parse_event(ev))
    return games_frame(rows)
