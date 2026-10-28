#!/usr/bin/env python3
"""Refresh first-quarter NBA player lines (stats.nba.com PlayerGameLogs, period 1) for this season and last.

One request per season returns every player's first quarter of every game, so a full refresh is cheap.
A failed season keeps the previously saved rows for it; the file is replaced only after a good read.
"""
from pathlib import Path
from datetime import date
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.refresh_nba_history import current_season
from scripts.stats_request import game_log

OUT = Path(__file__).resolve().parents[1] / 'data/nba/player_q1_logs.csv'
COLUMNS = ['SEASON', 'PLAYER_ID', 'PLAYER_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK']


def previous_season(season: str) -> str:
    first = int(season[:4]) - 1
    return f'{first}-{str(first + 1)[-2:]}'


def fetch(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs
    frames = []
    for season_type in ('Regular Season', 'Playoffs'):
        frame = game_log(playergamelogs.PlayerGameLogs, season_nullable=season, period_nullable=1,
                         season_type_nullable=season_type, league_id_nullable='00', timeout=60)
        if not frame.empty:
            frames.append(frame.assign(SEASON=season)[COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def main():
    old = pd.read_csv(OUT, dtype={'GAME_ID': str}) if OUT.exists() else pd.DataFrame(columns=COLUMNS)
    season = current_season(date.today())
    parts = []
    for name in (previous_season(season), season):
        try:
            parts.append(fetch(name))
        except Exception as exc:
            print(f'Q1 {name} unavailable ({exc}); keeping saved rows for it')
            parts.append(old[old['SEASON'] == name])
    result = pd.concat(parts, ignore_index=True)
    if result.empty:
        print('No first-quarter lines fetched; file unchanged.')
        return
    tmp = OUT.with_suffix('.csv.tmp')
    result.to_csv(tmp, index=False)
    tmp.replace(OUT)
    print(f'Q1 history: {len(result)} player-games across {result["GAME_ID"].nunique()} games')


if __name__ == '__main__':
    main()
