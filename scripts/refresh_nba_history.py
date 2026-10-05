#!/usr/bin/env python3
"""Refresh the current NBA season from the free NBA Stats game-log endpoint.

The default full-season read is necessary on clean CI runners: their checkout
does not contain the previous day's uncommitted box scores. Never downloads the
full Kaggle training archive or retrains saved model files.
"""
import argparse
from datetime import date, timedelta
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.refresh_wnba_history import save_history
from scripts.stats_request import game_log
from utils.wnba_loader import _normalize_player_logs


def current_season(today: date) -> str:
    first = today.year if today.month >= 10 else today.year - 1
    return f'{first}-{str(first + 1)[-2:]}'


def season_start(season: str) -> date:
    return date(int(season[:4]), 10, 1)


def fetch_recent(season: str, from_date: date) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog
    frames, failures = [], []
    for index, season_type in enumerate(('Regular Season', 'Playoffs', 'PlayIn', 'Pre Season')):
        if index:
            time.sleep(1)
        try:
            frame = game_log(leaguegamelog.LeagueGameLog,
                season=season, league_id='00', player_or_team_abbreviation='P',
                season_type_all_star=season_type,
                date_from_nullable=from_date.strftime('%m/%d/%Y'), timeout=30,
            )
            if not frame.empty:
                frame['SEASON'] = season
                frames.append(_normalize_player_logs(frame))
        except Exception as exc:
            failures.append(f'{season_type}: {exc}')
    if len(failures) == 4:
        raise RuntimeError('NBA Stats unavailable; existing history unchanged: ' + '; '.join(failures))
    if failures:
        print('Some NBA season types were unavailable: ' + '; '.join(failures))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', default=current_season(date.today()))
    parser.add_argument('--days', type=int, default=None,
                        help='Optional shorter local overlap; CI defaults to the full season')
    args = parser.parse_args()
    if args.days is not None and not 1 <= args.days <= 60:
        parser.error('--days must be between 1 and 60')
    from_date = (date.today() - timedelta(days=args.days)) if args.days is not None else season_start(args.season)
    incoming = fetch_recent(args.season, from_date)
    if incoming.empty:
        print('No new NBA box scores in the requested window; history unchanged.')
        return
    result, backup = save_history(incoming, Path(__file__).resolve().parents[1] / 'data/nba')
    print(f'NBA history through {pd.to_datetime(result.GAME_DATE, format="mixed").max().date()}; '
          f'{len(result)} rows. Backup: {backup}')


if __name__ == '__main__':
    main()
