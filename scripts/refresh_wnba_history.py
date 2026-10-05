#!/usr/bin/env python3
"""Refresh free WNBA game logs without retraining or deleting earlier seasons.

Usage: python3 scripts/refresh_wnba_history.py [--season 2026]
Includes regular season and playoffs; preserves backups before replacing files.
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import os
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from utils.wnba_loader import _normalize_player_logs
from utils.feature_engineering import engineer_features
from utils.pregame_features import prefer_identified_games
from scripts.stats_request import game_log


def merge_history(old, incoming):
    if incoming.empty:
        raise ValueError('Provider returned no games; keeping existing history')
    combined = pd.concat([old, incoming], ignore_index=True)
    combined['_day'] = pd.to_datetime(combined.GAME_DATE, format='mixed', errors='raise').dt.normalize()
    # Incoming full box scores supersede older copies of the same observation.
    combined = prefer_identified_games(combined)
    combined = combined.drop_duplicates(['PLAYER_NAME','_day'], keep='last').drop(columns='_day')
    return combined.sort_values(['PLAYER_NAME','GAME_DATE']).reset_index(drop=True)


def save_history(incoming, directory):
    directory = Path(directory)
    csv_path = directory/'player_game_logs.csv'
    parquet_path = directory/'engineered_data.parquet'
    old = pd.read_csv(csv_path)
    combined = merge_history(old, incoming)
    engineered = engineer_features(combined)
    engineered['_date'] = pd.to_datetime(engineered.GAME_DATE, format='mixed')
    # The provider returns some season/game identifiers as strings while the
    # saved CSV infers integers. Arrow cannot write a mixed object column.
    # IDs are labels rather than quantities, so keep one stable string type.
    for column in ('SEASON', 'SEASON_ID', 'Player_ID', 'TEAM_ID', 'Game_ID'):
        if column in engineered:
            engineered[column] = engineered[column].astype('string')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    csv_tmp = directory/f'.history-{stamp}.csv'
    parquet_tmp = directory/f'.history-{stamp}.parquet'
    try:
        combined.to_csv(csv_tmp,index=False)
        engineered.to_parquet(parquet_tmp,index=False)
    except Exception:
        csv_tmp.unlink(missing_ok=True)
        parquet_tmp.unlink(missing_ok=True)
        raise
    backup = directory/'backups'/stamp
    backup.mkdir(parents=True)
    for path in (csv_path, parquet_path):
        if path.exists():
            shutil.copy2(path, backup/path.name)
    os.replace(csv_tmp,csv_path)
    os.replace(parquet_tmp,parquet_path)
    return combined, backup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season',default=str(datetime.now().year))
    args=parser.parse_args()
    from nba_api.stats.endpoints import leaguegamelog
    frames=[]
    for index, season_type in enumerate(('Regular Season','Playoffs')):
        if index:
            time.sleep(1)
        raw=game_log(leaguegamelog.LeagueGameLog, season=args.season,league_id='10',player_or_team_abbreviation='P',season_type_all_star=season_type,timeout=20)
        if not raw.empty:
            raw['SEASON']=args.season
            frames.append(_normalize_player_logs(raw))
    if not frames:
        print('No WNBA box scores for this season yet; existing history unchanged.')
        return
    result,backup=save_history(pd.concat(frames,ignore_index=True),Path(__file__).resolve().parents[1]/'data/wnba')
    print(f'History through {pd.to_datetime(result.GAME_DATE,format="mixed").max().date()}; {len(result)} rows. Backup: {backup}')
    print('Restart the dashboard to reload history. This does not retrain models or establish betting performance.')


if __name__=='__main__':
    main()
