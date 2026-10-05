"""REST contracts without external services or paid odds calls."""
import math
import subprocess
import sys
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.routes import games, players, props, hitrates

client = TestClient(app)


def sample_history():
    return pd.DataFrame([
        dict(PLAYER_NAME='Test Player', GAME_DATE='Jan 02, 2025', _date=pd.Timestamp('2025-01-02'), PTS=10, REB=5, AST=2, TEAM_ABBREVIATION='NYL', SEASON='2025'),
        dict(PLAYER_NAME='Test Player', GAME_DATE='Dec 30, 2025', _date=pd.Timestamp('2025-12-30'), PTS=20, REB=10, AST=4, TEAM_ABBREVIATION='NYL', SEASON='2025'),
        dict(PLAYER_NAME='Test Player', GAME_DATE='Feb 01, 2026', _date=pd.Timestamp('2026-02-01'), PTS=30, REB=15, AST=6, TEAM_ABBREVIATION='NYL', SEASON='2026'),
    ])


def test_startup_does_not_import_dash_or_fetch():
    code = """
import socket
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network attempted'))
from fastapi.testclient import TestClient
from api.main import app
import sys
with TestClient(app) as client:
    assert client.get('/api/health').json() == {'status': 'ok'}
    assert client.get('/api/props').status_code == 200
    assert client.post('/api/props/refresh').json()['count'] == 0
assert 'dashboard.app' not in sys.modules
assert 'dash' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


@pytest.mark.parametrize('league', ['nba', 'wnba'])
def test_games_maps_espn_odds_and_constructs_matchup(monkeypatch, league):
    from utils import espn_game_odds, odds_fetcher
    monkeypatch.setattr(games, '_schedule', lambda league: ([{'HOME_TEAM': 'LAL', 'AWAY_TEAM': 'GSW'}, {'HOME_TEAM': 'NYK', 'AWAY_TEAM': 'BOS'}], '2026-09-28'))
    calls = []
    monkeypatch.setattr(espn_game_odds, 'get_game_odds', lambda selected, target: calls.append((selected, target)) or {'GSW@LAL': {'spread': {'home_line': -3.5}, 'total': {'line': 220.5}, 'h2h': {'home_price': -150, 'away_price': 130}, 'source': 'ESPN', 'bookmaker': 'ESPN BET', 'fetched_at': '2026-09-28T12:00:00+00:00'}, 'BOS@NYK': {'spread': None, 'total': None, 'h2h': None}})
    monkeypatch.setattr(odds_fetcher, 'get_game_odds', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('paid game odds called')))
    monkeypatch.setattr(games, '_team_injuries', lambda *_: [])
    response = client.get(f'/api/games?league={league}').json()['games']
    assert calls == [(league, '2026-09-28')]
    assert response[0]['matchup'] == 'GSW @ LAL'
    assert [response[0][k] for k in ['spread', 'total', 'home_ml', 'away_ml']] == [-3.5, 220.5, -150, 130]
    assert [response[0][k] for k in ['odds_source', 'odds_provider', 'odds_updated_at']] == ['ESPN', 'ESPN BET', '2026-09-28T12:00:00+00:00']
    assert response[1]['spread'] is None


def test_predictions_call_real_method_and_report_partial_failure(monkeypatch):
    from utils import game_predictor
    class Model:
        def __init__(self, *_): pass
        def predict_game(self, home, away, **kwargs):
            if home == 'BAD': raise ValueError('bad model input')
            return {'winner': home, 'predicted_spread': 5, 'predicted_total': 210, 'winner_confidence': 'MEDIUM'}
    monkeypatch.setattr(game_predictor, 'GamePredictor', Model)
    monkeypatch.setattr(games, 'history', lambda league: sample_history())
    monkeypatch.setattr(games.pd, 'read_csv', lambda *_: pd.DataFrame())
    monkeypatch.setattr(games, '_schedule', lambda league: ([{'HOME_TEAM': 'LAL', 'AWAY_TEAM': 'GSW'}, {'HOME_TEAM': 'BAD', 'AWAY_TEAM': 'GSW'}], '2026-09-28'))
    monkeypatch.setattr(games, '_cached_odds', lambda *_: {})
    monkeypatch.setattr(games, '_team_injuries', lambda *_: [])
    response = client.get('/api/games/predictions').json()
    assert response['predictions'][0]['spread'] == 5
    assert response['predictions'][0]['confidence'] == 'MEDIUM'
    assert len(response['errors']) == 1


def test_wnba_schedule_preserves_dash_game_details(monkeypatch):
    from utils import wnba_data_fetch
    monkeypatch.setattr(wnba_data_fetch, 'get_todays_wnba_games', lambda target: [
        {'game_id': '42', 'status': 'Scheduled', 'status_text': '7:00 pm ET', 'tip_time_et': '2026-09-28T19:00:00',
         'home': {'abbrev': 'NYL', 'name': 'New York Liberty', 'wins': 25, 'losses': 10, 'score': None},
         'away': {'abbrev': 'LVA', 'name': 'Las Vegas Aces', 'wins': 27, 'losses': 8, 'score': None}},
    ])
    monkeypatch.setattr(games, '_team_injuries', lambda league, team: [dict(name='Test Player', status='OUT', reason='Ankle')] if team == 'NYL' else [])
    monkeypatch.setattr(games, '_cached_odds', lambda *_: {})
    response = client.get('/api/games?league=wnba').json()
    game = response['games'][0]
    assert game['game_id'] == '42'
    assert game['status_text'] == '7:00 pm ET'
    assert (game['home_name'], game['home_wins'], game['home_losses']) == ('New York Liberty', 25, 10)
    assert game['home_injuries'] == [dict(name='Test Player', status='OUT', reason='Ankle')]


def test_wnba_hit_rates_use_original_matchup_computation(monkeypatch):
    from utils import wnba_data_fetch
    schedule = [{'home': {'abbrev': 'NYL'}, 'away': {'abbrev': 'LVA'}}]
    monkeypatch.setattr(wnba_data_fetch, 'get_todays_wnba_games', lambda target: schedule)
    rows = [dict(PLAYER_NAME='Test Player', TEAM_ABBREVIATION='NYL', _date=pd.Timestamp('2026-08-01') + pd.Timedelta(days=index), MIN=28, PTS=20, REB=8, AST=5, FG3M=2) for index in range(10)]
    monkeypatch.setattr(hitrates, 'history', lambda league: pd.DataFrame(rows))
    response = client.get('/api/wnba/hitrates').json()
    assert response['games'][0]['matchup'] == 'LVA @ NYL'
    assert any(entry['player_name'] == 'Test Player' and entry['stat'] == 'PTS' for entry in response['games'][0]['entries'])


def test_wnba_predictions_use_original_dash_predictor(monkeypatch):
    from utils import game_predictor
    class Model:
        def __init__(self, team_def_df, player_logs_df): pass
        def predict_game(self, home, away):
            assert (home, away) == ('NYL', 'LVA')
            return {'winner': 'NYL', 'predicted_spread': 3.2, 'predicted_total': 167.4,
                    'predicted_home_score': 85.3, 'predicted_away_score': 82.1,
                    'winner_confidence': 'LOW', 'intel': ['NYL recent form']}
    monkeypatch.setattr(game_predictor, 'GamePredictor', Model)
    monkeypatch.setattr(games, '_schedule', lambda league: ([{'HOME_TEAM': 'NYL', 'AWAY_TEAM': 'LVA'}], '2026-09-28'))
    monkeypatch.setattr(games, 'history', lambda league: sample_history())
    monkeypatch.setattr(games.pd, 'read_csv', lambda *_: pd.DataFrame())
    monkeypatch.setattr(games, '_cached_odds', lambda *_: {})
    response = client.get('/api/games/predictions?league=wnba').json()
    assert response['errors'] == []
    assert response['predictions'][0]['predicted_home_score'] == 85.3
    assert response['predictions'][0]['predicted_away_score'] == 82.1
    assert response['predictions'][0]['intel'] == ['NYL recent form']
    assert response['predictions'][0]['spread_pick'] is None
    assert response['predictions'][0]['total_pick'] is None


def test_game_market_picks_require_real_cached_lines(monkeypatch):
    from utils import game_predictor
    class Model:
        get_pick = game_predictor.GamePredictor.get_pick
        def __init__(self, *_): pass
        def predict_game(self, home, away, **kwargs):
            return {'winner': home, 'predicted_spread': 8, 'predicted_total': 230,
                    'winner_confidence': 'HIGH', 'predicted_home_score': 119,
                    'predicted_away_score': 111}
    monkeypatch.setattr(game_predictor, 'GamePredictor', Model)
    monkeypatch.setattr(games, '_schedule', lambda league: ([{'HOME_TEAM': 'LAL', 'AWAY_TEAM': 'GSW'}], '2026-09-28'))
    monkeypatch.setattr(games, 'history', lambda league: sample_history())
    monkeypatch.setattr(games.pd, 'read_csv', lambda *_: pd.DataFrame())
    monkeypatch.setattr(games, '_team_injuries', lambda *_: [])
    monkeypatch.setattr(games, '_cached_odds', lambda *_: {'GSW@LAL': {
        'spread': {'home_line': -3.5}, 'total': {'line': 220},
    }})
    prediction = client.get('/api/games/predictions').json()['predictions'][0]
    assert (prediction['spread_pick'], prediction['spread_team']) == ('HOME', 'LAL')
    assert prediction['total_pick'] == 'OVER'
    assert (prediction['market_spread'], prediction['market_total']) == (-3.5, 220)
    monkeypatch.setattr(games, '_cached_odds', lambda *_: {})
    without_market = client.get('/api/games/predictions').json()['predictions'][0]
    assert without_market['spread_pick'] is None
    assert without_market['total_pick'] is None


def test_game_lines_refresh_uses_espn_for_both_leagues_without_paid_provider(monkeypatch):
    from utils import espn_game_odds, odds_fetcher, odds_budget
    calls = []
    before = odds_budget.status()['daily']
    monkeypatch.setattr(odds_fetcher, 'get_game_odds', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('paid game odds called')))
    monkeypatch.setattr(espn_game_odds, 'get_game_odds', lambda league, target, force_refresh=False: calls.append((league, target, force_refresh)) or {'GSW@LAL': {}})
    for league in ('nba', 'wnba'):
        response = client.post(f'/api/games/refresh-lines?league={league}').json()
        assert response['count'] == 1 and response['source'] == 'ESPN'
    assert [(league, forced) for league, _, forced in calls] == [('nba', True), ('wnba', True)]
    assert odds_budget.status()['daily'] == before


def test_chart_dates_combo_under_and_push(monkeypatch):
    monkeypatch.setattr(players, 'history', lambda league: sample_history())
    response = client.get('/api/player/Test%20Player/chart-data?league=wnba&stat=PTS%2BREB&line=30&direction=under').json()
    assert [row['date'] for row in response['games']] == ['2025-01-02', '2025-12-30', '2026-02-01']
    assert [row['value'] for row in response['games']] == [15, 30, 45]
    assert [row['hit'] for row in response['games']] == [True, False, False]
    assert client.get('/api/player/Test%20Player/chart-data?stat=MIN').status_code == 400
    assert client.get('/api/player/Test%20Player/chart-data?line=NaN').status_code == 422


def test_stats_latest_season_and_actual_models(monkeypatch):
    monkeypatch.setattr(players, 'history', lambda league: sample_history())
    monkeypatch.setattr(players, '_injury_context', lambda *_: (None, None))
    class Model:
        def predict_player_game(self, name, frame, game_date):
            return {'predicted_pts': 23, 'predicted_reb': 8, 'predicted_ast': 5}
    calls = []
    def load(league, stat):
        calls.append((league, stat))
        return Model()
    monkeypatch.setattr(players, 'predictor', load)
    response = client.get('/api/player/Test%20Player/stats?league=wnba').json()
    assert response['season_avgs']['PTS'] == 30
    assert response['games_played'] == 1
    assert response['projections'] == {'PTS': 23, 'REB': 8, 'AST': 5}
    assert calls == [('wnba', 'PTS'), ('wnba', 'REB'), ('wnba', 'AST')]


def test_profile_uses_logged_player_id_and_actual_fg_percentage(monkeypatch):
    rows = sample_history()
    rows['Player_ID'] = [1631094, 1631094, 1631094]
    rows['FG_PCT'] = [0.5, 0.6, 0.45]
    monkeypatch.setattr(players, 'history', lambda league: rows)
    monkeypatch.setattr(players, 'predictor', lambda *_: None)
    monkeypatch.setattr(players, '_injury_context', lambda *_: ('QUESTIONABLE', 'Ankle'))
    response = client.get('/api/player/Test%20Player/stats?league=wnba').json()
    assert response['headshot_url'] == 'https://cdn.wnba.com/headshots/wnba/latest/1040x760/1631094.png'
    assert response['fg_pct'] == 45.0
    assert response['injury_status'] == 'QUESTIONABLE'
    assert response['injury_reason'] == 'Ankle'


def test_chart_supports_steals_blocks_and_season_venue(monkeypatch):
    rows = sample_history()
    rows['STL'] = [1, 2, 3]
    rows['BLK'] = [2, 1, 0]
    rows['MATCHUP'] = ['NYL @ LVA', 'NYL vs. LAS', 'NYL @ IND']
    monkeypatch.setattr(players, 'history', lambda league: rows)
    response = client.get('/api/player/Test%20Player/chart-data?league=wnba&stat=STL%2BBLK&games=164').json()
    assert [game['value'] for game in response['games']] == [3, 3, 3]
    assert [game['season'] for game in response['games']] == ['2025', '2025', '2026']
    assert [game['is_home'] for game in response['games']] == [False, True, False]


def test_unverified_props_never_claim_ev_or_locks(monkeypatch):
    from utils import odds_fetcher
    monkeypatch.setattr(odds_fetcher, '_cache', {})
    monkeypatch.setattr(props, '_get_props_data', lambda: {'main_page_data': [{'player': 'Test Player', 'stat': 'PTS', 'direction': 'Over', 'line': 20, 'hit_rate': .8, 'ev': .25, 'is_lock': True, 'has_live_odds': True, 'avg': math.nan}]})
    response = client.get('/api/props').json()
    assert response['props'] == [] and response['count'] == 0


def test_validation_and_wnba_unavailable_state():
    assert client.get('/api/players?league=football').status_code == 422
    assert client.get('/api/props?sort=bogus').status_code == 422
    response = client.get('/api/props?league=wnba').json()
    assert response['status'] == 'unavailable' and response['props'] == []


def test_exported_ui_routes_and_api_not_shadowed(tmp_path):
    import os
    for name in ('index', 'games', 'analysis', 'bets', '404'):
        (tmp_path / f'{name}.html').write_text(f'<h1>{name}</h1>')
    code = """
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
for name in ('games', 'analysis', 'bets'):
    response = client.get('/' + name)
    assert response.status_code == 200, response.text
    assert '<h1>' + name + '</h1>' in response.text
assert client.get('/api/health').json() == {'status': 'ok'}
assert client.get('/api/not-a-route').status_code == 404
"""
    env = dict(os.environ, SERVE_FRONTEND='1', FRONTEND_DIR=str(tmp_path))
    subprocess.run([sys.executable, '-c', code], check=True, timeout=30, env=env)


def test_local_history_infers_venue_and_reloads_when_file_changes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from api import data
    monkeypatch.setattr(data, 'get_config', lambda league: SimpleNamespace(data_dir=tmp_path))
    sample = sample_history()
    sample['MATCHUP'] = ['NYL @ LVA', 'NYL vs. LAS', 'NYL @ IND']
    path = tmp_path / 'player_game_logs.csv'
    sample.to_csv(path, index=False)
    frame = data.history('nba')
    assert frame['is_home'].tolist() == [0, 1, 0]
    sample.loc[0, 'PTS'] = 11
    sample.to_csv(path, index=False)
    assert data.history('nba').iloc[0]['PTS'] == 11


def test_refresh_evaluates_cached_quotes_without_fetching_and_rechecks_expiry(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    from api import data
    from utils import odds_fetcher
    now = datetime.now(timezone.utc)
    today = pd.Timestamp(now.astimezone(ZoneInfo('America/New_York')).date())
    rows = pd.DataFrame([dict(PLAYER_NAME='Test Player', GAME_DATE=(today-pd.Timedelta(days=i)).date().isoformat(),
                            _date=today-pd.Timedelta(days=i), PTS=30, TEAM_ABBREVIATION='LAL') for i in range(1, 13)])
    quote = dict(event_id='test', commence_time=(now+timedelta(minutes=1)).isoformat(), fetched_at=now.timestamp(),
                 updated_at=now.isoformat(), line=20.5, over_price=-110, under_price=-110)
    monkeypatch.setattr(odds_fetcher, '_cache', {'Test Player': {'PTS': quote}})
    monkeypatch.setattr(odds_fetcher, 'get_live_odds', lambda *a, **k: (_ for _ in ()).throw(AssertionError('paid fetch')))
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, 'history', lambda league: rows)
    class Model:
        calibration_residuals = [0.0]*100
        def predict_player_game(self, *args, **kwargs):
            return {'predicted_pts': 30}
    monkeypatch.setattr(data, 'predictor', lambda league, stat: Model())
    response = client.post('/api/props/refresh').json()
    assert response['count'] == 1
    prop = client.get('/api/props').json()['props'][0]
    assert prop['recommendation_eligible'] and prop['ev'] > 0
    assert prop['price'] == -110
    quote['fetched_at'] = now.timestamp() - 3600
    assert client.get('/api/props').json()['props'] == []


def test_refresh_empty_cache_is_explicit_and_never_fetches(monkeypatch):
    monkeypatch.setattr(props, '_cached_quotes', lambda league: {})
    monkeypatch.setattr(props, '_evaluated_cache', {})
    response = client.post('/api/props/refresh?league=wnba').json()
    assert response['count'] == 0
    assert 'no odds were fetched' in response['message'].lower()
    assert client.get('/api/props?league=wnba').json()['status'] == 'empty'
