import pandas as pd
from fastapi.testclient import TestClient
from api.main import app
from api.routes import trends as trends_route
from utils import trends

TODAY = pd.Timestamp('2026-03-10')


def frame(values, stat='REB', player='Big Man', team='AAA', minutes=30):
    """One player, `values` oldest to newest, one game per day ending the day before TODAY."""
    n = len(values)
    return pd.DataFrame({
        'PLAYER_NAME': player, 'TEAM_ABBREVIATION': team, 'MATCHUP': f'{team} vs. ZZZ', 'MIN': minutes,
        '_date': [TODAY - pd.Timedelta(days=n - i) for i in range(n)],
        **{s: (values if s == stat else [0] * n) for s in ('PTS', 'REB', 'AST', 'FG3M')}})


def cards(df, **kw):
    return trends.build_trends(df, today=TODAY, **kw)


def test_derived_under_card_uses_tightest_line_with_enough_hits():
    # nine of ten games below 5.5, one outlier
    result = [c for c in cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4])) if c['side'] == 'under']
    assert len(result) == 1
    card = result[0]
    assert (card['line'], card['hits'], card['games'], card['stat']) == (5.5, 9, 10, 'REB')
    assert card['results'] == [True] * 8 + [False, True]
    assert card['text'] == 'Under 5.5 rebounds in 9 of last 10'
    assert card['stars'] == 4 and card['live'] is None and card['line_source'] == 'derived'


def test_derived_over_card_and_trivial_lines_skipped():
    over = [c for c in cards(frame([9, 10, 11, 12, 9, 10, 8, 11, 2, 10], stat='PTS')) if c['side'] == 'over']
    assert over[0]['line'] == 7.5 and over[0]['hits'] == 9
    # a bench player who grabs 1 rebound a night never yields an "Under 3.5" headline
    assert cards(frame([0, 1, 0, 1, 0, 1, 0, 0, 1, 0])) == []


def test_live_quote_sets_line_price_and_side():
    quote = {'line': 6.5, 'over_price': -110, 'under_price': -125, 'bookmaker': 'DK',
             'home_team': 'AAA', 'away_team': 'BBB'}
    card = cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4]), quotes={'Big Man': {'REB': quote}})[0]
    assert (card['side'], card['line'], card['line_source']) == ('under', 6.5, 'live')
    assert card['live'] == {'line': 6.5, 'price': -125, 'book': 'DK'}
    assert card['opponent'] == 'BBB'


def test_too_few_games_stale_or_low_minutes_players_are_skipped():
    assert cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 3])) == []
    assert cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 3, 4]), recent_days=0) == []
    assert cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 3, 4], minutes=5)) == []


def test_nan_values_never_count_as_hits_and_cards_sort_by_stars():
    assert cards(frame([float('nan')] * 10)) == []
    both = pd.concat([frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4], player='A'),
                      frame([2, 3, 4, 5, 3, 4, 2, 5, 3, 4], player='B')])
    result = cards(both)
    assert [c['stars'] for c in result] == sorted([c['stars'] for c in result], reverse=True)
    assert (result[0]['player'], result[0]['stars']) == ('A', 4) and {c['player'] for c in result} == {'A', 'B'}


def test_route_limits_and_validates(monkeypatch):
    df = frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4])
    monkeypatch.setattr(trends_route, 'history', lambda league: df)
    monkeypatch.setattr(trends_route, 'live_quotes', lambda league: {})
    monkeypatch.setattr(trends_route, 'today', lambda: TODAY)
    client = TestClient(app)
    body = client.get('/api/trends?league=nba&limit=1').json()
    assert body['league'] == 'nba' and body['count'] == 1 and body['trends'][0]['player'] == 'Big Man'
    assert client.get('/api/trends?league=nba&limit=0').status_code == 422
