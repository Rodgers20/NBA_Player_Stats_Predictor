"""Early-season trust in a team-history forecast.

The game model only sees box-score history, so in October it rates a team by last
season's roster. Until a team has played enough games this season, pull the
forecast margin toward a neutral home-court baseline instead of trusting it.
"""
from __future__ import annotations

from math import erf, sqrt

import pandas as pd

# Games at which the model is trusted half-way. ponytail: a single constant; fit it
# per-league from season-opening backtests if early-season Brier needs tuning.
TRUST_HALF_LIFE_GAMES = 12


def current_season_games(history: pd.DataFrame, team: str, target: pd.Timestamp, league: str = 'nba') -> int | None:
    """Regular-season games ``team`` played in the season containing ``target``; None if history can't say."""
    if not {'team', 'date'}.issubset(history.columns):
        return None
    # NBA seasons straddle New Year (July rollover); WNBA seasons sit inside one year.
    start_year = target.year - int(league == 'nba' and target.month < 7)
    start = pd.Timestamp(year=start_year, month=7 if league == 'nba' else 1, day=1)
    rows = history[(history.team == team) & (history.date >= start) & (history.date < target)]
    if 'playoffs' in rows:
        rows = rows[rows.playoffs == 0]
    return len(rows)


def shrink_early_season(forecast: dict, games: float, baseline: float) -> dict:
    """Return a copy of ``forecast`` with margin, win chance and interval pulled toward ``baseline``.

    ``games`` is the smaller current-season game count of the two teams.
    """
    weight = max(games, 0) / (max(games, 0) + TRUST_HALF_LIFE_GAMES)
    margin = baseline + (forecast['margin'] - baseline) * weight
    shift = margin - forecast['margin']
    out = dict(forecast, margin=margin, early_season_weight=weight)
    sigma = float(forecast.get('margin_sd') or 0)
    if sigma > 0:
        out['home_win_probability'] = 0.5 * (1 + erf(margin / (sigma * sqrt(2))))
    intervals = forecast.get('prediction_intervals')
    if intervals:
        out['prediction_intervals'] = dict(
            intervals, margin_80=[v + shift for v in intervals.get('margin_80', [])])
    return out
