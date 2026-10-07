"""WNBA Props separates unpriced research from a five-player priced board."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api import data
from api.routes import props
from utils import wnba_data_fetch, wnba_odds_fetcher

client = TestClient(app)
ET = ZoneInfo('America/New_York')


def fresh_history(names):
    today = pd.Timestamp(datetime.now(ET).date())
    return pd.DataFrame([
        dict(PLAYER_NAME=name, GAME_DATE=(today - pd.Timedelta(days=day)).date().isoformat(),
             _date=today - pd.Timedelta(days=day), TEAM_ABBREVIATION='NYL',
             MIN=30, PTS=20 + index, REB=8 + index, AST=4 + index)
        for index, name in enumerate(names) for day in range(1, 13)
    ])


def mock_wnba_model(monkeypatch):
    class Model:
        calibration_residuals = [0.0] * 100

        def predict_player_game(self, *args, **kwargs):
            return {'predicted_pts': 30, 'predicted_reb': 15, 'predicted_ast': 9}

    monkeypatch.setattr(data, 'predictor', lambda league, stat: Model())


def reject_paid_fetch(monkeypatch):
    monkeypatch.setattr(wnba_odds_fetcher, 'get_live_wnba_odds',
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Odds API called on page load')))


def test_wnba_research_uses_fresh_slate_and_model_without_market_claims(monkeypatch):
    names = [f'Player {index}' for index in range(7)]
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, 'history', lambda league: fresh_history(names))
    monkeypatch.setattr(wnba_data_fetch, 'get_todays_wnba_games', lambda target: [
        {'home': {'abbrev': 'NYL'}, 'away': {'abbrev': 'LVA'}}])
    mock_wnba_model(monkeypatch)
    reject_paid_fetch(monkeypatch)

    response = client.get('/api/props?league=wnba&stat=PTS').json()
    assert response['status'] == 'research'
    assert response['game_matchups'] == ['LVA @ NYL']
    assert response['count'] == len(response['props']) == 7
    assert len({row['player'] for row in response['props']}) == 7
    for row in response['props']:
        assert row['model_projection'] == 30
        assert row['line'] is None and row['live_line'] is None
        assert row['price'] is None and row['ev'] is None and row['model_prob'] is None
        assert row['recommendation_eligible'] is False and row['has_live_odds'] is False
        assert row['direction'] == 'Research'


def test_wnba_research_without_stat_covers_every_stat(monkeypatch):
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, 'history', lambda league: fresh_history([f'Player {i}' for i in range(7)]))
    monkeypatch.setattr(wnba_data_fetch, 'get_todays_wnba_games', lambda target: [
        {'home': {'abbrev': 'NYL'}, 'away': {'abbrev': 'LVA'}}])
    mock_wnba_model(monkeypatch)

    response = client.get('/api/props?league=wnba').json()
    assert response['stat_counts'] == {'PTS': 7, 'REB': 7, 'AST': 7}
    assert response['count'] == 21


def test_wnba_priced_board_caps_board_at_fifteen_distinct_players(monkeypatch):
    from utils import market_evaluation

    names = [f'Player {index}' for index in range(20)]
    rows = fresh_history(names)
    now = datetime.now(timezone.utc)
    start = datetime.combine(now.astimezone(ET).date(), datetime.max.time(), ET)
    quote = dict(event_id='event', commence_time=start.isoformat(),
                 fetched_at=now.timestamp(), updated_at=now.isoformat(), line=20.5,
                 over_price=-110, under_price=-110, home_team='NYL', away_team='LVA',
                 game_matchup='LVA @ NYL')
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, 'history', lambda league: rows)
    monkeypatch.setattr(wnba_odds_fetcher, '_cache', {
        name: {'PTS': dict(quote), 'REB': dict(quote)} for name in names})
    mock_wnba_model(monkeypatch)
    monkeypatch.setattr(market_evaluation, 'evaluate_market',
                        lambda projection, line, price, residuals, direction:
                        {'ev': 0.2 if direction == 'Over' else -0.1,
                         'model_prob': 0.75, 'probability_source': 'Test model'})
    reject_paid_fetch(monkeypatch)

    refreshed = client.post('/api/props/refresh?league=wnba').json()
    assert refreshed['count'] == 15
    assert refreshed['status'] == 'ready'
    board = client.get('/api/props?league=wnba').json()
    assert board['status'] == 'ready'
    assert board['count'] == len(board['props']) == 15
    assert len({row['player'] for row in board['props']}) == 15
    assert all(row['ev'] > 0 and row['recommendation_eligible'] for row in board['props'])
