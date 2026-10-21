"""Trend cards from local game logs; live prices come only from already-cached quotes."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from fastapi import APIRouter, Query
from api.data import League, history
from utils import trends
from utils.prop_quality import quote_problem

router = APIRouter(tags=['trends'])


def today():
    return pd.Timestamp(datetime.now(ZoneInfo('America/New_York')).date())


def live_quotes(league):
    """Fresh cached quotes only ({player: {stat: quote}}); never triggers an odds fetch."""
    from api.routes.props import _cached_quotes
    try:
        quotes = _cached_quotes(league)
    except Exception as exc:  # trends still work without prices
        print(f'[trends] cached quotes unavailable: {exc}')
        return {}
    return {player: {stat: quote for stat, quote in markets.items() if not quote_problem(quote)}
            for player, markets in quotes.items()}


def slate_matchups(league):
    """(date, {team: opponent}) for the next slate; (None, {}) when the schedule provider is down."""
    from api.routes.games import _schedule
    try:
        records, target = _schedule(league)
    except Exception as exc:  # trends still work without knowing who plays tonight
        print(f'[trends] schedule unavailable: {exc}')
        return None, {}
    teams = {}
    for row in records:
        home, away = row.get('HOME_TEAM'), row.get('AWAY_TEAM')
        if home and away:
            teams[home], teams[away] = away, home
    return target, teams


@router.get('/trends')
def get_trends(league: League = 'nba', limit: int = Query(60, ge=1, le=500)):
    slate_date, teams = slate_matchups(league)
    cards = trends.build_trends(history(league), live_quotes(league), today(), slate=teams or None)[:limit]
    return dict(league=league, count=len(cards), trends=cards,
                slate=dict(date=slate_date, teams=sorted(teams)))
