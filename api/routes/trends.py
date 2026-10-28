"""Trend cards from local game logs; live prices come only from already-cached quotes."""
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from typing import Literal
from fastapi import APIRouter, Query
from api.data import League, history, with_q1
from utils import trends
from utils.prop_quality import quote_problem
from utils.quarter_stats import Q1_MIN_HITS, Q1_TREND_STATS

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
def get_trends(league: League = 'nba', limit: int = Query(60, ge=1, le=500), period: Literal['game', '1q'] = 'game'):
    slate_date, teams = slate_matchups(league)
    from api.routes.props import _headshot_urls
    if period == '1q':  # no first-quarter prices are bought, so every 1Q line is derived from the games
        cards = trends.build_trends(with_q1(history(league), league), {}, today(), slate=teams or None,
                                    stats=Q1_TREND_STATS, min_hits=Q1_MIN_HITS)[:limit]
    else:
        cards = trends.build_trends(history(league), live_quotes(league), today(), slate=teams or None)[:limit]
    headshots = _headshot_urls(league) if cards else {}
    cards = [dict(card, headshot_url=headshots.get(card['player'])) for card in cards]
    return dict(league=league, period=period, count=len(cards), trends=cards,
                slate=dict(date=slate_date, teams=sorted(teams)))
