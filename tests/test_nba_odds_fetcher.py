"""Regression coverage for slate identity and quote cache expiry."""
import pytest
from utils import odds_fetcher as odds


def event(event_id, starts, line):
    return {'id': event_id, 'commence_time': starts,
            'home_team': 'Boston Celtics', 'away_team': 'Miami Heat',
            'bookmakers': [{'key': 'fanduel', 'markets': [
                {'key': 'player_points', 'outcomes': [
                    {'description': 'A', 'name': side, 'point': line, 'price': -110}
                    for side in ('Over', 'Under')]}]}]}


def test_target_event_wins_regardless_of_payload_order():
    earlier = event('earlier', '2026-09-17T00:00:00Z', 12.5)
    target = event('target', '2026-09-18T00:00:00Z', 22.5)
    for events in ((earlier, target), (target, earlier)):
        result = {}
        for payload in events:
            odds._parse_event_odds(payload, result, target_date='2026-09-17')
        quote = result['A']['PTS']
        assert quote['line'] == 22.5
        assert quote['event_id'] == 'target'
        assert quote['event_date'] == '2026-09-17'  # US evening, next UTC day
        assert quote['home_team'] == 'BOS' and quote['away_team'] == 'MIA'


@pytest.mark.parametrize('field,value', [('id', None), ('commence_time', 'bad'),
                                        ('commence_time', '2026-09-17T20:00:00'),
                                        ('home_team', 'unknown')])
def test_incomplete_event_is_not_actionable(field, value):
    payload = event('target', '2026-09-18T00:00:00Z', 22.5)
    payload[field] = value
    result = {}
    odds._parse_event_odds(payload, result, target_date='2026-09-17')
    assert result == {}


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('age,same_date,usable', [(60, True, True), (1800, True, False),
                                               (3600, True, False), (60, False, False)])
def test_fallback_requires_fresh_cache_for_same_slate(monkeypatch, disabled, age, same_date, usable):
    cached = {'A': {'PTS': {'line': 22.5}}}
    monkeypatch.setattr(odds, 'API_KEY', 'test')
    monkeypatch.setattr(odds, '_cache', cached)
    monkeypatch.setattr(odds, '_cache_ts', 10000 - age)
    monkeypatch.setattr(odds, '_cache_date', '2026-09-17' if same_date else '2026-09-16')
    monkeypatch.setattr(odds, '_player_props_unavailable', disabled)
    monkeypatch.setattr(odds.time, 'time', lambda: 10000)
    def fail(*args):
        raise RuntimeError('offline')
    monkeypatch.setattr(odds, '_fetch_event_ids', fail)
    assert odds.get_live_odds(force_refresh=True, target_date='2026-09-17') == (cached if usable else {})


def test_fetch_filters_slate_and_records_cache_date(monkeypatch):
    monkeypatch.setattr(odds, 'API_KEY', 'test')
    monkeypatch.setattr(odds, '_cache', {})
    monkeypatch.setattr(odds, '_cache_ts', 0)
    monkeypatch.setattr(odds, '_cache_date', None)
    monkeypatch.setattr(odds, '_player_props_unavailable', False)
    monkeypatch.setattr(odds, '_fetch_event_ids', lambda *args: ['earlier', 'target'])
    payloads = {'earlier': event('earlier', '2026-09-17T00:00:00Z', 12.5),
                'target': event('target', '2026-09-18T00:00:00Z', 22.5)}
    monkeypatch.setattr(odds, '_fetch_event_odds', lambda event_id, markets: payloads[event_id])
    monkeypatch.setattr(odds.time, 'sleep', lambda seconds: None)
    result = odds.get_live_odds(force_refresh=True, target_date='2026-09-17')
    assert result['A']['PTS']['event_id'] == 'target'
    assert odds._cache_date == '2026-09-17'
