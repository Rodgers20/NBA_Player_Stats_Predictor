"""Local-only data and model access; importing the API never starts Dash."""
from functools import lru_cache
from typing import Literal
import math
import pandas as pd
from fastapi import HTTPException
from utils.league_config import get_config

League = Literal['nba', 'wnba']


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def _load_history(league, modified):
    path = get_config(league).data_dir / 'player_game_logs.csv'
    df = pd.read_csv(path, low_memory=False)
    df['_date'] = pd.to_datetime(df['GAME_DATE'], format='mixed', errors='coerce')
    df = df.dropna(subset=['_date', 'PLAYER_NAME'])
    from utils.pregame_features import prefer_identified_games
    df = prefer_identified_games(df).drop_duplicates(['PLAYER_NAME', '_date'])
    if 'TEAM_ABBREVIATION' not in df:
        df['TEAM_ABBREVIATION'] = df['MATCHUP'].str.split().str[0]
    if 'is_home' not in df:
        df['is_home'] = df['MATCHUP'].str.contains('vs.', regex=False, na=False).astype(int)
    return df.sort_values('_date')


def history(league):
    path = get_config(league).data_dir / 'player_game_logs.csv'
    if not path.exists():
        raise HTTPException(503, f'{league.upper()} player history has not been exported')
    try:
        return _load_history(league, path.stat().st_mtime_ns)
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(503, f'{league.upper()} player history is unavailable') from exc


@lru_cache(maxsize=12)
def _load_model(league, stat, modified):
    from models.predictor import StatPredictor
    return StatPredictor.load(str(get_config(league).models_dir / f'{stat.lower()}_predictor.pkl'))


def predictor(league, stat):
    path = get_config(league).models_dir / f'{stat.lower()}_predictor.pkl'
    return _load_model(league, stat, path.stat().st_mtime_ns) if path.exists() else None
