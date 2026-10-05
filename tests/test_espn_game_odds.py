"""ESPN game lines remain source-labelled, slate-scoped, and optional."""
import pytest

from utils import espn_game_odds as odds


@pytest.fixture(autouse=True)
def clear_cache():
    odds._CACHE.clear()
    yield
    odds._CACHE.clear()


def event(home='LAL', away='GSW', lines=None, event_id='401', competition_id='501'):
    return {
        'id': event_id,
        'competitions': [{
            'id': competition_id,
            'competitors': [
                {'homeAway': 'home', 'team': {'abbreviation': home}},
                {'homeAway': 'away', 'team': {'abbreviation': away}},
            ],
            'odds': lines or [],
        }],
    }


def line(spread=-4.5, total=224.5, home_ml=-180, away_ml=155, provider='ESPN BET'):
    return {
        'provider': {'name': provider, 'priority': 3},
        'spread': spread,
        'overUnder': total,
        'homeTeamOdds': {'moneyLine': home_ml},
        'awayTeamOdds': {'moneyLine': away_ml},
    }


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_scoreboard_parses_all_markets_and_source_without_core_request(monkeypatch):
    calls = []
    monkeypatch.setattr(odds.requests, 'get',
                        lambda url, **kw: calls.append((url, kw)) or Response({'events': [event(lines=[line()])]}))
    result = odds.get_game_odds('nba', '2026-10-02')
    quote = result['GSW@LAL']
    assert quote['spread'] == {'home_line': -4.5}
    assert quote['total'] == {'line': 224.5}
    assert quote['h2h'] == {'home_price': -180.0, 'away_price': 155.0}
    assert (quote['source'], quote['bookmaker']) == ('ESPN', 'ESPN BET')
    assert quote['fetched_at']
    assert len(calls) == 1
    assert calls[0][1]['params'] == {'dates': '20261002'}


def test_away_favorite_spread_is_flipped_to_home_perspective():
    marked = line(spread=-3.5)
    marked['details'] = 'GSW -3.5'
    result = odds.get_game_odds('nba', '2026-10-02', [event(lines=[marked])])
    assert result['GSW@LAL']['spread']['home_line'] == 3.5


def test_missing_scoreboard_market_uses_core_fallback(monkeypatch):
    calls = []
    def fake_get(url, **kw):
        calls.append(url)
        assert '/events/401/competitions/501/odds' in url
        return Response({'items': [line(spread=-2.5, total=219.5)]})
    monkeypatch.setattr(odds.requests, 'get', fake_get)
    result = odds.get_game_odds('wnba', '2026-10-02', [event(home='NYL', away='LV')])
    assert result['LVA@NYL']['spread'] == {'home_line': -2.5}
    assert result['LVA@NYL']['total'] == {'line': 219.5}
    assert len(calls) == 1
    assert '/leagues/wnba/' in calls[0]


def test_cache_is_isolated_by_league_and_eastern_slate_date(monkeypatch):
    monkeypatch.setattr(odds, '_fetch_core', lambda *args: None)
    first = odds.get_game_odds('nba', '2026-10-02', [event(lines=[line(spread=-4.5)])])
    cached = odds.get_game_odds('nba', '2026-10-02', [event(lines=[line(spread=-9.5)])])
    wnba = odds.get_game_odds('wnba', '2026-10-02', [event(home='NYL', away='LV', lines=[line(spread=-6.5)])])
    next_date = odds.get_game_odds('nba', '2026-10-03', [event(lines=[line(spread=-1.5)])])
    assert first['GSW@LAL']['spread']['home_line'] == -4.5
    assert cached['GSW@LAL']['spread']['home_line'] == -4.5
    assert wnba['LVA@NYL']['spread']['home_line'] == -6.5
    assert next_date['GSW@LAL']['spread']['home_line'] == -1.5


def test_missing_markets_stay_missing_and_projection_providers_are_excluded(monkeypatch):
    monkeypatch.setattr(odds, '_fetch_core', lambda *args: None)
    partial = {'provider': {'name': 'ESPN BET'}, 'spread': -3.5}
    projection = line(provider='NumberFire')
    result = odds.get_game_odds('nba', '2026-10-02', [event(lines=[projection, partial])])
    assert result['GSW@LAL']['spread'] == {'home_line': -3.5}
    assert result['GSW@LAL']['total'] is None
    assert result['GSW@LAL']['h2h'] is None
    empty = odds.get_game_odds('nba', '2026-10-03', [event(lines=[projection])])
    assert empty == {}


def test_wnba_valkyries_and_sparks_match_local_team_codes(monkeypatch):
    monkeypatch.setattr(odds, '_fetch_core', lambda *args: None)
    result = odds.get_game_odds('wnba', '2026-10-02', [
        event(home='GS', away='DAL', lines=[line()]),
        event(home='LAS', away='NYL', lines=[line()]),
    ])
    assert 'DAL@GSV' in result
    assert 'NYL@LAS' in result
