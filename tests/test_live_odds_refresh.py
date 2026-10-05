"""Explicit refresh budget, provenance and board constraints; no real network."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api import data
from api.routes import props
from utils import odds_budget, odds_fetcher as nba, wnba_odds_fetcher as wnba


@pytest.mark.parametrize('module,fn,home,away', [
    (nba, 'get_live_odds', 'Los Angeles Lakers', 'Golden State Warriors'),
    (wnba, 'get_live_wnba_odds', 'Las Vegas Aces', 'Indiana Fever'),
])
def test_explicit_refresh_limits_events_markets_and_preserves_metadata(monkeypatch, module, fn, home, away):
    now = datetime.now(timezone.utc)
    events = [dict(id=str(i), commence_time=(now+timedelta(seconds=i+1)).isoformat(), home_team=home, away_team=away) for i in range(4)]
    events += [dict(id='past', commence_time=(now-timedelta(hours=1)).isoformat()),
               dict(id='tomorrow', commence_time=(now+timedelta(days=1)).isoformat())]
    calls = []
    class Response:
        status_code = 200
        headers = {}
        def __init__(self, payload): self.payload = payload
        def json(self): return self.payload
    def get(url, params, timeout):
        calls.append((url, params))
        if url.endswith('/events'):
            return Response(events)
        event = next(e for e in events if f"/events/{e['id']}/" in url)
        return Response(dict(event, bookmakers=[dict(key='fanduel', last_update=now.isoformat(), markets=[
            dict(key='player_points', outcomes=[dict(description='Player '+event['id'], name='Over', point=20.5, price=-110)])])]))
    monkeypatch.setattr(module, 'API_KEY', 'fake')
    monkeypatch.setattr(module, '_cache', {})
    monkeypatch.setattr(module, '_cache_date', None)
    monkeypatch.setattr(module.requests, 'get', get)
    getter = getattr(module, fn)
    assert getter() == {} and calls == []
    result = getter(force_refresh=True)
    assert len(result) == 2 and len(calls) == 3
    assert all(call[1]['markets'] == 'player_points,player_rebounds,player_assists' for call in calls[1:])
    quote = result['Player 0']['PTS']
    assert quote['event_id'] == '0'
    assert quote['updated_at'] == now.isoformat() and quote['fetched_at'] > 0
    assert quote['event_date'] == now.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    assert odds_budget.status()['daily'] == 6
    getter()
    assert len(calls) == 3


def test_api_refresh_is_explicit_and_board_preserves_each_player_stat(monkeypatch):
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ZoneInfo('America/New_York')).date())
    quote = dict(event_id='test', commence_time=(now+timedelta(minutes=1)).isoformat(), fetched_at=now.timestamp(),
                 updated_at=now.isoformat(), line=20.5, over_price=-110, under_price=-110, home_team='LAL', away_team='GSW')
    cached = {f'Player {i}': {'PTS': dict(quote), 'REB': dict(quote)} for i in range(7)}
    rows = pd.DataFrame([dict(PLAYER_NAME=player, GAME_DATE=(today-pd.Timedelta(days=day)).date().isoformat(),
                             _date=today-pd.Timedelta(days=day), PTS=30, REB=30, TEAM_ABBREVIATION='LAL')
                         for player in cached for day in range(1, 13)])
    calls = []
    def fetch(force_refresh=False):
        calls.append(force_refresh)
        monkeypatch.setattr(nba, '_cache', cached)
        return cached
    monkeypatch.setattr(nba, '_cache', {})
    monkeypatch.setattr(nba, 'get_live_odds', fetch)
    monkeypatch.setattr(props, 'history', lambda league: rows)
    monkeypatch.setattr(props, '_evaluated_cache', {})
    class Model:
        calibration_residuals = [0.0]*100
        def predict_player_game(self, *a, **kw): return {'predicted_pts': 30, 'predicted_reb': 30}
    monkeypatch.setattr(data, 'predictor', lambda *a: Model())
    client = TestClient(app)
    assert client.get('/api/props/budget').status_code == 200
    assert client.post('/api/props/refresh').json()['count'] == 0 and calls == []
    response = client.post('/api/props/refresh?fetch_odds=true').json()
    assert calls == [True] and response['count'] == 14
    assert response['status'] == 'ready' and response['budget']['max_refresh_cost'] == 6
    board = client.get('/api/props?direction=all').json()['props']
    assert len(board) == 14
    assert len({(p['player'], p['stat']) for p in board}) == 14
    assert all(p['ev'] > 0 and p['recommendation_eligible'] for p in board)
