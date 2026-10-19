"""Exact date schedules must never silently become a different slate."""
import pytest
from fastapi import HTTPException
from api.routes import games


@pytest.mark.parametrize('value', ['2026-02-30', '20261008', '2026-1-08', 'nope'])
def test_invalid_dates_rejected_before_provider(monkeypatch, value):
    monkeypatch.setattr(games, '_schedule', lambda *args: pytest.fail('provider called'))
    for endpoint in (games.get_games, games.get_predictions):
        with pytest.raises(HTTPException) as error:
            endpoint('nba', value)
        assert error.value.status_code == 422


def test_empty_requested_date_is_not_next_slate(monkeypatch):
    calls = []
    monkeypatch.setattr(games, '_schedule', lambda *args: (calls.append(args) or [], args[1]))
    for endpoint in (games.get_games, games.get_predictions):
        assert endpoint('wnba', '2099-10-08')['target_date'] == '2099-10-08'
    assert calls == [('wnba', '2099-10-08'), ('wnba', '2099-10-08')]


@pytest.mark.parametrize('state,score', [('pre', None), ('in', '42'), ('post', '42')])
def test_scoreboard_status_and_actual_scores(monkeypatch, state, score):
    from utils import basketball_content
    calls = []
    def fetch(league, path, **params):
        calls.append((league, path, params))
        return {'events': [{'id': '123', 'date': '2026-10-09T00:00:00Z',
            'status': {'type': {'state': state, 'shortDetail': 'Final' if state == 'post' else 'Scheduled'}},
            'competitions': [{'venue': {'fullName': 'Arena'}, 'broadcasts': [{'names': ['ESPN']}],
                'competitors': [{'homeAway': side, 'score': '42',
                    'team': {'abbreviation': abbr, 'displayName': abbr, 'logo': 'https://logo', 'color': '000000'},
                    'records': [{'type': 'total', 'summary': '20-10'}]}
                    for side, abbr in [('home', 'LV'), ('away', 'NY')]]}]}]}
    monkeypatch.setattr(basketball_content, '_fetch', fetch)
    rows, target = games._schedule('wnba', '2026-10-08')
    assert target == '2026-10-08'
    assert calls == [('wnba', 'scoreboard', {'dates': '20261008'})]
    assert rows[0]['HOME_TEAM'] == 'LVA'
    assert rows[0]['HOME_SCORE'] == score
    assert rows[0]['GAME_STATUS_STATE'] == state
    assert rows[0]['HOME_LOGO'] == 'https://logo'
    assert rows[0]['HOME_WINS'] == '20'
    assert rows[0]['BROADCAST'] == 'ESPN'


def test_historical_date_withholds_current_model_predictions(monkeypatch):
    monkeypatch.setattr(games, '_schedule', lambda *args: pytest.fail('historical projection attempted'))
    response = games.get_predictions('nba', '2000-01-01')
    assert response['target_date'] == '2000-01-01'
    assert response['predictions'] == []
    assert response['errors'] == []
    assert 'historical forecast' in response['message']
