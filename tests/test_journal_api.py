"""Exercise REST boundaries against an isolated database, never personal data."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.journal import router
from utils import odds_budget


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(odds_budget, 'DB_PATH', tmp_path / 'journal.sqlite3')
    app = FastAPI()
    app.include_router(router, prefix='/api')
    with TestClient(app) as client:
        yield client


def entry(**changes):
    return dict(dict(token='one-entry', mode='paper', league='wnba', player="A'ja Wilson",
                     game_date='2026-09-28', stat='PTS', side='Over', line=24.5,
                     price=-110, book='Test book', stake=11, notes=''), **changes)


def test_save_is_idempotent_and_modes_are_separate(client):
    first = client.post('/api/journal', json=entry())
    assert first.status_code == 200
    assert client.post('/api/journal', json=entry()).json() == first.json()
    client.post('/api/journal', json=entry(token='real-entry', mode='real', stake=22))
    paper = client.get('/api/journal?mode=paper').json()
    real = client.get('/api/journal?mode=real').json()
    assert len(paper['bets']) == len(real['bets']) == 1
    assert paper['summary']['pending'] == 11
    assert real['summary']['pending'] == 22
    assert paper['summary']['roi'] is None


def test_settlement_corrections_keep_existing_accounting(client):
    identifier = client.post('/api/journal', json=entry()).json()['id']
    for result, profit, settled, pending in [
        ('win', 10, 1, 0), ('loss', -11, 1, 0), ('push', 0, 1, 0),
        ('void', 0, 0, 0), ('pending', 0, 0, 11),
    ]:
        response = client.patch(f'/api/journal/{identifier}', json={'result': result})
        assert response.status_code == 200
        report = client.get('/api/journal').json()['summary']
        assert (report['profit'], report['settled'], report['pending']) == (profit, settled, pending)
        assert report['staked'] == (11 if settled else 0)
        assert (report['roi'] is None) == (settled == 0)


@pytest.mark.parametrize('changes', [
    {'token': ''}, {'book': ' '}, {'player': ''}, {'stake': 0}, {'stake': -1},
    {'line': 1.3}, {'price': 0}, {'game_date': 'bad'}, {'mode': 'automatic'},
    {'league': 'other'}, {'stat': 'TRIPLE_DOUBLE'}, {'price': 'Infinity'},
])
def test_invalid_entries_do_not_write(client, changes):
    assert client.post('/api/journal', json=entry(**changes)).status_code == 422
    assert client.get('/api/journal').json()['bets'] == []


@pytest.mark.parametrize('stat', ['FG3M', 'STL', 'BLK', 'PTS+REB', 'PTS+AST', 'REB+AST', 'PTS+REB+AST', 'STL+BLK'])
def test_supported_prop_stats_can_be_tracked(client, stat):
    response = client.post('/api/journal', json=entry(stat=stat))
    assert response.status_code == 200
    assert client.get('/api/journal').json()['bets'][0]['stat'] == stat


def test_invalid_modes_and_settlements(client):
    assert client.get('/api/journal?mode=invalid').status_code == 422
    assert client.patch('/api/journal/missing', json={'result': 'win'}).status_code == 404
    assert client.patch('/api/journal/missing', json={'result': 'wrong'}).status_code == 422


def test_public_container_disables_private_journal(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    monkeypatch.setenv('SERVE_FRONTEND', '1')
    monkeypatch.delenv('JOURNAL_PRIVATE', raising=False)
    response = TestClient(app).get('/api/journal')
    assert response.status_code == 503
    assert 'local app' in response.json()['detail']
