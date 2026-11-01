"""Team game logs: records going into a game, last-10 form (overall and home-or-road split), head-to-head.

Every lookup is "strictly before this tip-off", so a finished game's page shows what was knowable then.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl

FORM_GAMES = 10
H2H_LIMIT = 10


@dataclass(frozen=True)
class Played:
    game_id: str
    season: int
    kickoff: datetime
    date: str
    opp_id: str
    opp: str
    opp_abbr: str
    home: bool       # the team was the designated home team on a non-neutral floor (False on neutral floors)
    neutral: bool
    conf_game: bool
    pts_for: int
    pts_against: int

    @property
    def win(self) -> bool:
        return self.pts_for > self.pts_against


def _entry(p: Played) -> dict[str, Any]:
    return {"date": p.date, "opp": p.opp, "opp_abbr": p.opp_abbr, "home": p.home, "win": p.win,
            "pts_for": p.pts_for, "pts_against": p.pts_against}


def summarize(games: list[Played]) -> dict[str, Any]:
    n = len(games)
    wins = sum(p.win for p in games)
    pf = sum(p.pts_for for p in games)
    pa = sum(p.pts_against for p in games)
    return {"games": [_entry(p) for p in games], "n": n, "wins": wins, "losses": n - wins,
            "record": f"{wins}-{n - wins}",
            "avg_scored": round(pf / n, 1) if n else None, "avg_allowed": round(pa / n, 1) if n else None,
            "avg_margin": round((pf - pa) / n, 1) if n else None}


class TeamLogs:
    """Chronological played games per team, built once from the final scores."""

    def __init__(self, finals: pl.DataFrame) -> None:
        by_team: dict[str, list[Played]] = {}
        for g in finals.sort("kickoff", "game_id").to_dicts():
            for side, other in (("home", "away"), ("away", "home")):
                p = Played(
                    g["game_id"], g["season"], g["kickoff"], g["gameday"], g[f"{other}_id"],
                    g[f"{other}_short"] or g[f"{other}_name"] or "", g[f"{other}_abbr"] or "", side == "home" and not g["neutral"],
                    bool(g["neutral"]), bool(g["conf_game"]), g[f"{side}_score"], g[f"{other}_score"])
                by_team.setdefault(g[f"{side}_id"], []).append(p)
        self._games = by_team
        self._kickoffs = {t: [p.kickoff for p in ps] for t, ps in by_team.items()}

    def before(self, team: str, kickoff: datetime) -> list[Played]:
        games = self._games.get(team, [])
        return games[:bisect_left(self._kickoffs.get(team, []), kickoff)]

    def form(self, team: str, kickoff: datetime, *, at_home: bool | None = None, n: int = FORM_GAMES) -> dict[str, Any]:
        """Last ``n`` games oldest to newest. ``at_home`` True/False keeps only home/road games (neutral ones drop)."""
        past = self.before(team, kickoff)
        if at_home is not None:
            past = [p for p in past if not p.neutral and p.home == at_home]
        return summarize(past[-n:])

    def record(self, team: str, kickoff: datetime, season: int, *, conference: bool = False) -> str:
        past = [p for p in self.before(team, kickoff) if p.season == season and (p.conf_game or not conference)]
        wins = sum(p.win for p in past)
        return f"{wins}-{len(past) - wins}"

    def streak(self, team: str, kickoff: datetime, season: int) -> str | None:
        past = [p for p in self.before(team, kickoff) if p.season == season]
        if not past:
            return None
        last, count = past[-1].win, 0
        for p in reversed(past):
            if p.win != last:
                break
            count += 1
        return f"{'W' if last else 'L'}{count}"

    def head_to_head(self, home_id: str, away_id: str, kickoff: datetime) -> dict[str, Any]:
        """Meetings before ``kickoff`` since the first ingested season, newest first, from the home team's view."""
        meetings = [p for p in self.before(home_id, kickoff) if p.opp_id == away_id]
        wins = sum(p.win for p in meetings)
        n = len(meetings)
        return {"n": n, "home_wins": wins, "away_wins": n - wins,
                "avg_margin_home": round(sum(p.pts_for - p.pts_against for p in meetings) / n, 1) if n else None,
                "games": [{"date": p.date, "game_id": p.game_id, "neutral": p.neutral, "at_home": p.home,
                           "home_team_pts": p.pts_for, "away_team_pts": p.pts_against,
                           "winner": "home" if p.win else "away"} for p in reversed(meetings[-H2H_LIMIT:])]}
