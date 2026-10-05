from fastapi.testclient import TestClient
from api.main import app
from utils import scheduled_players as source


def event(when='2026-10-05T01:00:00Z', status='STATUS_SCHEDULED'):
    return {'date': when, 'status': {'type': {'name': status}}, 'competitions': [{'competitors': [
        {'team': {'id': '1', 'abbreviation': 'NY'}},
        {'team': {'id': '2', 'abbreviation': 'LA'}},
    ]}]}


def test_only_scheduled_team_rosters_and_reported_out_excluded(monkeypatch):
    calls = []
    def fetch(league, path, params=None):
        calls.append((league, path, params))
        if path == 'scoreboard':
            return {'events': [event(), event('2026-10-06T01:00:00Z')]}
        return {'athletes': [
            {'displayName': path, 'status': {'type': 'active'}},
            {'displayName': 'Out player', 'injuries': [{'status': 'Out'}]},
            {'displayName': 'Inactive player', 'status': {'type': 'inactive'}},
        ]}
    monkeypatch.setattr(source, '_fetch', fetch)
    result = source.scheduled_players('wnba', '2026-10-04')
    assert result['game_date'] == '2026-10-04'
    assert len(result['players']) == 2
    assert {p['team'] for p in result['players']} == {'NYL', 'LAS'}
    assert all(call[0] == 'wnba' for call in calls)
    assert calls[0][2] == {'dates': '20261004'}


def test_empty_wrong_day_or_cancelled_schedule_never_loads_rosters(monkeypatch):
    for events in ([], [event('2026-10-06T01:00:00Z')], [event(status='STATUS_POSTPONED')]):
        calls = []
        def fetch(league, path, params=None):
            calls.append(path)
            return {'events': events}
        monkeypatch.setattr(source, '_fetch', fetch)
        result = source.scheduled_players('nba', '2026-10-04')
        assert result['players'] == []
        assert calls == ['scoreboard']


def test_provider_failure_does_not_offer_partial_or_historical_roster(monkeypatch):
    import requests
    def fetch(*args, **kwargs):
        raise requests.Timeout()
    monkeypatch.setattr(source, '_fetch', fetch)
    assert source.scheduled_players('nba', '2026-10-04')['players'] == []


def test_api_validates_date_and_league_and_forwards_scope(monkeypatch):
    monkeypatch.setattr(source, 'scheduled_players', lambda league, day: dict(league=league, game_date=day, players=[]))
    client = TestClient(app)
    assert client.get('/api/players/scheduled?league=wnba&game_date=2026-10-04').json() == {
        'league': 'wnba', 'game_date': '2026-10-04', 'players': []}
    assert client.get('/api/players/scheduled?league=mlb&game_date=2026-10-04').status_code == 422
    assert client.get('/api/players/scheduled?league=nba&game_date=bad').status_code == 422
