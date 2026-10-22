"""Evaluated prop rows carry line-shopping, edge, hit-window, teammate and matchup fields."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import data
from api.main import app
from api.routes import props
from utils import defense_rank, odds_budget, wnba_odds_fetcher

client = TestClient(app)
ET = ZoneInfo('America/New_York')


def history_frame():
    today = pd.Timestamp(datetime.now(ET).date())
    rows = []
    for day in range(1, 16):
        stamp = today - pd.Timedelta(days=day)
        opponent = 'LVA' if day % 3 == 0 else 'ATL'
        base = dict(GAME_DATE=stamp.date().isoformat(), _date=stamp, TEAM_ABBREVIATION='NYL',
                    MATCHUP=f'NYL vs. {opponent}', SEASON='2026', MIN=30)
        without_b = day > 10                       # teammate B missed days 11-15
        rows.append(dict(base, PLAYER_NAME='Player A', PTS=26 if without_b else 18, REB=5, AST=3))
        if not without_b:
            rows.append(dict(base, PLAYER_NAME='Player B', PTS=10, REB=5, AST=3))
    return pd.DataFrame(rows)


@pytest.fixture
def refreshed(monkeypatch, tmp_path):
    monkeypatch.setattr(odds_budget, 'DB_PATH', tmp_path / 'p.sqlite3')
    now = datetime.now(timezone.utc)
    start = datetime.combine(now.astimezone(ET).date(), datetime.max.time(), ET)
    books = [dict(book='FanDuel', line=22.5, over_price=-110, under_price=-110),
             dict(book='DraftKings', line=21.5, over_price=-105, under_price=-125)]
    quote = dict(event_id='e', commence_time=start.isoformat(), fetched_at=now.timestamp(),
                 updated_at=now.isoformat(), line=22.5, over_price=-110, under_price=-110,
                 bookmaker='FanDuel', books=books, home_team='NYL', away_team='LVA', game_matchup='LVA @ NYL')

    class Model:
        calibration_residuals = [-4.0, -2.0, 2.0, 4.0] * 25

        def predict_player_game(self, *args, **kwargs):
            return {'predicted_pts': 26, 'predicted_reb': 5, 'predicted_ast': 3}

    frame = history_frame()
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, 'history', lambda league: frame)
    monkeypatch.setattr(data, 'predictor', lambda league, stat: Model())
    monkeypatch.setattr(wnba_odds_fetcher, '_cache', {'Player A': {'PTS': quote}})
    monkeypatch.setattr(props, '_injuries_for', lambda league, team: [
        dict(name='Player B', status='OUT'), dict(name='Player A', status='OUT'),
        dict(name='Healthy', status='PROBABLE')])
    monkeypatch.setattr(defense_rank, 'matchup',
                        lambda league, player, opponent, stat: dict(def_rank=3, def_rank_of=13, matchup_grade='A'))
    assert client.post('/api/props/refresh?league=wnba').json()['count'] == 1
    return client.get('/api/props?league=wnba').json()['props'][0]


def test_line_shopping_fields(refreshed):
    assert [b['book'] for b in refreshed['books']] == ['FanDuel', 'DraftKings']
    assert refreshed['best_over'] == dict(book='DraftKings', price=-105, line=21.5)
    assert refreshed['best_under'] == dict(book='FanDuel', price=-110, line=22.5)
    assert refreshed['consensus']['line'] == 21.5      # tie between two lines goes to the lower
    assert refreshed['open_line'] is None or isinstance(refreshed['open_line'], float)


def test_edge_and_matchup_fields(refreshed):
    assert refreshed['direction'] == 'Over' and refreshed['edge_pct'] > 0
    assert refreshed['stars'] in (1, 2, 3, 4, 5) and isinstance(refreshed['high_edge'], bool)
    assert 0 < refreshed['implied_prob'] < 1
    assert (refreshed['def_rank'], refreshed['def_rank_of'], refreshed['matchup_grade']) == (3, 13, 'A')


def test_hit_windows_use_pre_game_history(refreshed):
    windows = refreshed['hit_windows']
    assert windows['L5'] == dict(hits=0, total=5, pct=0.0)           # days 1-5: 18 pts vs 22.5
    assert windows['L20']['total'] == 15 and windows['L20']['hits'] == 5
    assert windows['season'] == windows['L20']
    assert windows['h2h'] == dict(hits=2, total=5, pct=40.0)         # LVA days 3, 6, 9 (18 pts) and 12, 15 (26 pts)


def test_teammate_split_lists_only_key_out_teammates(refreshed):
    assert refreshed['teammate_splits'] == [dict(
        teammate='Player B', status='OUT', avg_with=18.0, avg_without=26.0, games_without=5,
        hit_rate_without=100.0)]


def test_serialised_prop_carries_the_last_ten_results_newest_first():
    row = props._serialise_prop(dict(player='X', stat='PTS', l10_values=list(range(12))), None)
    assert row['l10_values'] == list(range(10))


def test_serialised_prop_without_results_has_an_empty_list():
    assert props._serialise_prop(dict(player='X', stat='PTS'), None)['l10_values'] == []


def test_priced_fields_empty_for_research_rows():
    row = props._serialise_prop(dict(player='X', stat='PTS'), 'No verified sportsbook line')
    assert row['stars'] is None and row['edge_pct'] is None and row['high_edge'] is False
    assert row['books'] == [] and row['teammate_splits'] == [] and row['hit_windows'] is None
