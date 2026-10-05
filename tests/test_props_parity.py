"""The REST prop board preserves the Dash app's available rows and filters."""

from fastapi.testclient import TestClient

from api.main import app
from api.routes import props


client = TestClient(app)


def _row(player, stat, direction, home, ev):
    return dict(player=player, stat=stat, direction=direction, team='LAL' if home else 'GSW',
                opponent='GSW' if home else 'LAL', game_matchup='GSW @ LAL',
                is_home_today=home, line=20.5, avg=25, hit_rate=.7, ev=ev,
                model_prob=.72, probability_source='Test model',
                has_live_odds=True, live_over_price=-110, live_under_price=-110,
                l5_values=[20, 22, 23, 24, 25])


def test_board_keeps_multiple_stats_and_filters_by_venue(monkeypatch):
    cache = dict(target_date='2026-10-01', game_matchups=['GSW @ LAL'], main_page_data=[
        _row('Player A', 'PTS', 'Over', True, .12),
        _row('Player A', 'REB', 'Under', True, .10),
        _row('Player B', 'PTS', 'Over', False, .09),
    ])
    monkeypatch.setattr(props, '_get_props_data', lambda: cache)
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, '_quality_reason', lambda *args: None)

    board = client.get('/api/props?direction=all').json()
    assert board['count'] == 3
    assert board['stat_counts'] == {'PTS': 2, 'REB': 1}
    assert [(item['player'], item['stat']) for item in board['props']].count(('Player A', 'REB')) == 1
    assert client.get('/api/props?direction=all&location=home').json()['count'] == 2
    assert client.get('/api/props?direction=all&location=away').json()['count'] == 1
    assert client.get('/api/props?direction=all&stat=REB').json()['count'] == 1
    assert client.get('/api/props?direction=all&limit=1').json()['count'] == 3


def test_research_rows_never_claim_price_or_expected_value(monkeypatch):
    cache = dict(target_date='2026-10-01', game_matchups=['GSW @ LAL'], main_page_data=[
        dict(_row('Player A', 'PTS', 'Over', True, .12), chart_windows={'l5': {'values': [1, 2], 'labels': ['a', 'b']}})
    ])
    monkeypatch.setattr(props, '_get_props_data', lambda: cache)
    monkeypatch.setattr(props, '_evaluated_cache', {})
    monkeypatch.setattr(props, '_quality_reason', lambda *args: 'Sportsbook price is unavailable')

    assert client.get('/api/props').json()['props'] == []
    board = client.get('/api/props?include_research=true').json()
    assert board['count'] == 1
    item = board['props'][0]
    assert item['quality_reason'] == 'Sportsbook price is unavailable'
    assert item['recommendation_eligible'] is False
    assert item['ev'] is None and item['price'] is None and item['model_prob'] is None
    assert item['has_live_odds'] is False and item['is_lock'] is False
    assert item['chart_windows']['l5']['values'] == [1, 2]


def test_supporting_props_sections_are_available_without_fake_bet_claims(monkeypatch):
    monkeypatch.setattr(props, '_get_props_data', lambda: {
        'target_date': '2026-10-01', 'alt_lines_data': [
            dict(team='LAL', player='Player A', stat='PTS', stat_label='Points',
                 threshold=20, trend='5/L5', price=-110, ev=.2)
        ]})
    from utils import prediction_tracker
    monkeypatch.setattr(prediction_tracker, 'get_props_record', lambda: {'hit': 4, 'total': 5, 'pct': 80})

    alt = client.get('/api/props/alt-lines').json()
    assert alt['count'] == 1 and alt['recommendation_eligible'] is False
    assert 'price' not in alt['alt_lines'][0] and 'ev' not in alt['alt_lines'][0]
    assert client.get('/api/props/record').json()['total'] == 5
    parlays = client.get('/api/props/parlays').json()
    assert parlays['recommendation_eligible'] is False and parlays['total_count'] == 0
    assert 'over' in parlays['sections']
