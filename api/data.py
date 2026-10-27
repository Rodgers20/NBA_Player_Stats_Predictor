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


REAL_SEASON_PREFIXES = ('2', '4', '5')   # nba_api SEASON_ID: 1 preseason, 2 regular, 3 All-Star, 4 playoffs, 5 play-in


def played_games(df):
    """Only real-season games the player took the floor in.

    Preseason minutes and did-not-play rows (blank or zero minutes, zero stats) dragged every
    average, hit rate and projection toward zero: 18% of NBA rows were DNPs."""
    keep = pd.Series(True, index=df.index)
    if 'SEASON_ID' in df:
        keep &= df['SEASON_ID'].astype(str).str.startswith(REAL_SEASON_PREFIXES)
    if 'MIN' in df:
        minutes = pd.to_numeric(df['MIN'], errors='coerce')
        stats = df[[c for c in ('PTS', 'REB', 'AST') if c in df]].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
        keep &= (minutes > 0) | (minutes.isna() & (stats > 0))
    return df[keep]


@lru_cache(maxsize=4)
def _load_history(league, modified):
    path = get_config(league).data_dir / 'player_game_logs.csv'
    df = played_games(pd.read_csv(path, low_memory=False))
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


def q1_lines(league):
    """First-quarter player lines (scripts/refresh_q1_history.py); empty when none have been fetched."""
    path = get_config(league).data_dir / 'player_q1_logs.csv'
    return _load_q1(path, path.stat().st_mtime_ns) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=2)
def _load_q1(path, modified):
    return pd.read_csv(path, dtype={'GAME_ID': str})


def with_q1(frame, league):
    """`frame` (history or a slice of it) plus PTS_1Q/REB_1Q/AST_1Q/FG3M_1Q."""
    from utils.quarter_stats import with_q1 as join
    return join(frame, q1_lines(league))


@lru_cache(maxsize=12)
def _load_model(league, stat, modified):
    from models.predictor import StatPredictor
    return StatPredictor.load(str(get_config(league).models_dir / f'{stat.lower()}_predictor.pkl'))


def predictor(league, stat):
    if '+' in stat:  # a combo market is priced from its single-stat models
        from utils.combo_model import ComboModel
        parts = {part: predictor(league, part) for part in stat.split('+')}
        return None if any(model is None for model in parts.values()) else ComboModel(parts)
    path = get_config(league).models_dir / f'{stat.lower()}_predictor.pkl'
    return _load_model(league, stat, path.stat().st_mtime_ns) if path.exists() else None
