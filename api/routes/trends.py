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


@router.get('/trends')
def get_trends(league: League = 'nba', limit: int = Query(60, ge=1, le=500)):
    cards = trends.build_trends(history(league), live_quotes(league), today())[:limit]
    return dict(league=league, count=len(cards), trends=cards)
