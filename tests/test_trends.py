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


def test_derived_lines_never_produce_unders():
    # Books only offer unders on the posted line; an alternate line can only be bet over.
    # nine of ten games below 5.5 would make a derived "Under 5.5", which cannot be bet.
    result = cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4]))
    assert result and all(c['side'] == 'over' for c in result if c['line_source'] == 'derived')


def test_posted_line_keeps_the_under_card_with_its_price():
    quote = {'line': 5.5, 'over_price': -110, 'under_price': -125, 'bookmaker': 'DK',
             'home_team': 'AAA', 'away_team': 'BBB'}
    result = cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4]), quotes={'Big Man': {'REB': quote}})
    assert [(c['side'], c['line'], c['hits'], c['line_source']) for c in result] == [('under', 5.5, 9, 'live')]
    assert result[0]['text'] == 'Under 5.5 rebounds in 9 of last 10' and result[0]['stars'] == 4


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
    assert {c['player'] for c in result} == {'A', 'B'}


def test_slate_marks_players_playing_today_and_ranks_them_first():
    streak = [2, 3, 4, 5, 3, 4, 2, 5, 3, 4]
    both = pd.concat([frame(streak, player='Away Guy', team='AAA'), frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4], player='Idle Star', team='BBB')])
    result = cards(both, slate={'AAA': 'CCC'})
    assert result[0]['player'] == 'Away Guy' and result[0]['playing_today'] is True and result[0]['opponent'] == 'CCC'
    idle = [c for c in result if c['player'] == 'Idle Star']
    assert idle and all(c['playing_today'] is False for c in idle)
    assert all(c['playing_today'] is None for c in cards(both))  # no schedule: unknown, not "not playing"


def test_route_reports_the_slate(monkeypatch):
    df = frame([2, 3, 4, 5, 3, 4, 2, 5, 3, 4])
    monkeypatch.setattr(trends_route, 'history', lambda league: df)
    monkeypatch.setattr(trends_route, 'live_quotes', lambda league: {})
    monkeypatch.setattr(trends_route, 'today', lambda: TODAY)
    monkeypatch.setattr(trends_route, 'slate_matchups', lambda league: ('2026-03-10', {'AAA': 'CCC', 'CCC': 'AAA'}))
    body = TestClient(app).get('/api/trends?league=nba').json()
    assert body['slate'] == {'date': '2026-03-10', 'teams': ['AAA', 'CCC']}
    assert body['trends'][0]['playing_today'] is True


def test_route_limits_and_validates(monkeypatch):
    monkeypatch.setattr(trends_route, 'slate_matchups', lambda league: (None, {}))
    df = frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4])
    monkeypatch.setattr(trends_route, 'history', lambda league: df)
    monkeypatch.setattr(trends_route, 'live_quotes', lambda league: {})
    monkeypatch.setattr(trends_route, 'today', lambda: TODAY)
    client = TestClient(app)
    body = client.get('/api/trends?league=nba&limit=1').json()
    assert body['league'] == 'nba' and body['count'] == 1 and body['trends'][0]['player'] == 'Big Man'
    assert client.get('/api/trends?league=nba&limit=0').status_code == 422


def frame_with(**columns):
    n = 10
    base = frame([0] * n)
    return base.assign(**{key: value for key, value in columns.items()})


def test_combos_steals_and_blocks_get_cards_but_never_combo_unders():
    df = frame_with(PTS=[20, 22, 25, 21, 24, 23, 26, 22, 25, 24], REB=[8, 9, 10, 9, 8, 10, 9, 8, 9, 10],
                    AST=[5, 6, 7, 5, 6, 7, 6, 5, 6, 7], STL=[2, 2, 2, 3, 2, 2, 2, 2, 1, 2], BLK=[2, 2, 2, 3, 2, 2, 2, 2, 1, 2])
    result = cards(df)
    stats = {c['stat'] for c in result}
    assert {'PTS+REB', 'PTS+AST', 'REB+AST', 'PTS+REB+AST', 'STL', 'BLK'} <= stats
    assert all(c['side'] == 'over' for c in result if '+' in c['stat'])
    pra = next(c for c in result if c['stat'] == 'PTS+REB+AST')
    assert pra['avg'] == 38.2 and pra['text'].startswith('Over ') and 'points + rebounds + assists' in pra['text']


def test_a_posted_combo_line_is_over_only_even_when_the_under_would_hit():
    df = frame_with(PTS=[10] * 10, REB=[5] * 10, AST=[4] * 10, STL=[0] * 10, BLK=[0] * 10)
    quote = {'line': 25.5, 'over_price': -110, 'under_price': -110, 'bookmaker': 'DK', 'home_team': 'AAA', 'away_team': 'BBB'}
    result = cards(df, quotes={'Big Man': {'PTS+REB+AST': quote}})
    assert [c for c in result if c['stat'] == 'PTS+REB+AST' and c['line_source'] == 'live'] == []


def test_equal_streaks_rank_by_cushion_not_by_name():
    # Both clear their line 10 of 10; the one whose average sits further above it ranks first, A-Z be damned.
    thin = frame([8] * 10, stat='PTS', player='Aaron Thin')
    wide = frame([20] * 10, stat='PTS', player='Zed Wide')
    quotes = {'Aaron Thin': {'PTS': {'line': 7.5, 'over_price': -110}}, 'Zed Wide': {'PTS': {'line': 12.5, 'over_price': -110}}}
    order = [c['player'] for c in cards(pd.concat([thin, wide]), quotes=quotes) if c['side'] == 'over']
    assert order[:2] == ['Zed Wide', 'Aaron Thin']


def test_cards_carry_each_game_value_for_the_chart():
    card = next(c for c in cards(frame([2, 3, 4, 5, 3, 4, 2, 5, 9, 4])) if c['side'] == 'over')
    assert card['values'] == [2, 3, 4, 5, 3, 4, 2, 5, 9, 4]
