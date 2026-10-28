"""Per-game context for a player's log: minutes, team margin, result, days of rest and which key teammates sat out.

The player logs cover one team's tracked players in most games, so a game's score comes from the team-game history
rather than from summing player points.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import pandas as pd

ROTATION_MINUTES = 20.0   # a teammate averaging at least this many minutes is a rotation player
ROTATION_GAMES = 5        # ... over at least this many games with the team that season
MAX_OUT = 6               # most absent teammates listed per game
MAX_REST_DAYS = 10        # a longer gap is an off-season, not rest
_TEAM_ALIAS = {'SAN': 'SAS'}   # player logs say SAN; team history says SAS


class LeagueContext(NamedTuple):
    margins: dict    # (game id, team) -> points for minus points against
    rotation: dict   # (season, team) -> rotation player names, most minutes first
    present: dict    # (game id, team) -> frozenset of names who played


def _team(matchup) -> str:
    parts = str(matchup or '').split()
    return parts[0] if parts else ''


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def build_league_context(history: pd.DataFrame, teams: pd.DataFrame | None) -> LeagueContext:
    """Index the whole league once; `annotate` then answers per player without rescanning."""
    frame = history.assign(
        _team=history['MATCHUP'].map(_team),
        _gid=pd.to_numeric(history['Game_ID'], errors='coerce'),
        _minutes=pd.to_numeric(history['MIN'], errors='coerce'),
    ).dropna(subset=['_gid'])
    present = {(int(gid), team): frozenset(names)
               for (gid, team), names in frame.groupby(['_gid', '_team'])['PLAYER_NAME'].agg(list).items()}

    per_player = frame.groupby(['SEASON', '_team', 'PLAYER_NAME'])['_minutes'].agg(['mean', 'count']).reset_index()
    regulars = per_player[(per_player['mean'] >= ROTATION_MINUTES) & (per_player['count'] >= ROTATION_GAMES)]
    regulars = regulars.sort_values('mean', ascending=False)
    rotation = regulars.groupby(['SEASON', '_team'])['PLAYER_NAME'].agg(list).to_dict()

    margins = {}
    if teams is not None and len(teams):
        ids = pd.to_numeric(teams['GAME_ID'], errors='coerce')
        for game_id, team, scored, allowed in zip(ids, teams['TEAM_ABBREVIATION'], teams['PTS'], teams['OPP_PTS']):
            if pd.notna(game_id) and pd.notna(scored) and pd.notna(allowed):
                margins[(int(game_id), team)] = int(scored) - int(allowed)
    return LeagueContext(margins, rotation, present)


def annotate(player_games: pd.DataFrame, context: LeagueContext) -> list[dict]:
    """One record per row of `player_games` (oldest first): minutes, margin, won, rest and out."""
    records, previous = [], None
    for _, row in player_games.iterrows():
        team = _team(row.get('MATCHUP'))
        game_id = pd.to_numeric(row.get('Game_ID'), errors='coerce')
        known = pd.notna(game_id)
        rest = None
        if previous is not None:
            days = (row['_date'] - previous).days - 1
            rest = int(days) if 0 <= days <= MAX_REST_DAYS else None
        previous = row['_date']
        result = str(row.get('WL') or '')
        played = context.present.get((int(game_id), team), frozenset()) if known else frozenset()
        out = [name for name in context.rotation.get((row.get('SEASON'), team), [])
               if name not in played and name != row['PLAYER_NAME']][:MAX_OUT] if known else []
        records.append(dict(
            minutes=_number(row.get('MIN')),
            margin=context.margins.get((int(game_id), _TEAM_ALIAS.get(team, team))) if known else None,
            won=True if result == 'W' else False if result == 'L' else None,
            rest=rest,
            out=out,
        ))
    return records
