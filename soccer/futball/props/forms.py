"""Turn `player_match_stats` rows into per-player form and per-team aggregates, optionally as of a date."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class PlayerForm:
    espn_id: str
    player: str
    team: str
    competition_code: str
    position: str | None
    games: int            # appearances (starts + substitute appearances)
    starts: int
    subs: int
    team_games: int       # matches his team has played in the window (any competition)
    last5_starts: int     # starts in the team's last five matches
    last5_games: int      # how many of those five exist
    gap: int              # team matches since his last appearance
    goals: int
    assists: int
    shots: int
    sot: int


@dataclass(frozen=True)
class TeamAggregate:
    games: int
    goals_for: int
    goals_against: int
    sot_for: int
    sot_against: int


def _in_window(rows: list[dict], before: str | None) -> list[dict]:
    return [r for r in rows if before is None or r["date"] < before]


def _team_matches(rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """team -> its matches as (date, event_id), oldest first."""
    seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        seen[r["team"]].add((r["date"], r["event_id"]))
    return {team: sorted(matches) for team, matches in seen.items()}


def team_aggregates(rows: list[dict], before: str | None = None) -> dict[str, TeamAggregate]:
    window = _in_window(rows, before)
    goals: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)  # team -> event -> (for, against)
    sot_for: Counter = Counter()
    sot_against: Counter = Counter()
    for r in window:
        goals[r["team"]][r["event_id"]] = (r["team_goals"], r["opp_goals"])
        sot_for[r["team"]] += r["sot"]
        sot_against[r["opponent"]] += r["sot"]
    return {
        team: TeamAggregate(
            games=len(events), goals_for=sum(f for f, _ in events.values()),
            goals_against=sum(a for _, a in events.values()),
            sot_for=sot_for[team], sot_against=sot_against[team],
        )
        for team, events in goals.items()
    }


def build_forms(rows: list[dict], before: str | None = None) -> list[PlayerForm]:
    """One PlayerForm per player, using only matches strictly before `before` (None = all)."""
    window = _in_window(rows, before)
    matches = _team_matches(window)
    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in window:
        by_player[r["espn_id"]].append(r)

    forms: list[PlayerForm] = []
    for pid, player_rows in by_player.items():
        latest = max(player_rows, key=lambda r: (r["date"], r["event_id"]))
        team = latest["team"]
        mine = [r for r in player_rows if r["team"] == team]  # a transfer resets his record
        team_matches = matches[team]
        last5 = team_matches[-5:]
        last5_events = {event for _, event in last5}
        played = {r["event_id"] for r in mine}
        last_idx = max(i for i, (_, event) in enumerate(team_matches) if event in played)
        domestic = Counter(r["competition_code"] for r in mine if r["competition_code"] != "UCL")
        competition = (domestic or Counter(r["competition_code"] for r in mine)).most_common(1)[0][0]
        position = Counter(r["position"] for r in mine if r["position"]).most_common(1)
        starts = sum(1 for r in mine if r["started"])
        forms.append(PlayerForm(
            espn_id=pid, player=latest["player"], team=team, competition_code=competition,
            position=position[0][0] if position else None,
            games=len(mine), starts=starts, subs=len(mine) - starts,
            team_games=len(team_matches),
            last5_starts=sum(1 for r in mine if r["started"] and r["event_id"] in last5_events),
            last5_games=len(last5), gap=len(team_matches) - 1 - last_idx,
            goals=sum(r["goals"] for r in mine), assists=sum(r["assists"] for r in mine),
            shots=sum(r["shots"] for r in mine), sot=sum(r["sot"] for r in mine),
        ))
    return forms
