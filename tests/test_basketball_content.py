from utils import basketball_content as content


def event(game_id='1', completed=True, score=90, opponent_score=80):
    return {'id': game_id, 'date': '2026-06-01T23:00Z', 'seasonType': {'type': 2},
            'competitions': [{'status': {'type': {'completed': completed}}, 'competitors': [
                {'id': '20', 'homeAway': 'home', 'score': {'value': score}},
                {'id': '8', 'score': {'value': opponent_score}, 'team': {'abbreviation': 'MIN'}}]}]}


def test_team_real_scores_and_season(monkeypatch):
    calls = []
    def fetch(league, path, **params):
        calls.append((path, params))
        if path == 'teams':
            return {'sports': [{'leagues': [{'season': {'year': 2026}, 'teams': [
                {'team': {'id': '20', 'abbreviation': 'ATL', 'displayName': 'Atlanta Dream'}}]}]}]}
        playoff = event('playoff'); playoff['seasonType']['type'] = 3
        return {'team': {'groups': {'id': '1'}}, 'events': [event(), event('future', False), playoff]}
    monkeypatch.setattr(content, '_fetch', fetch)
    payload = content.get_teams('wnba', 2025)
    team = payload['teams'][0]
    assert payload['season'] == 2025
    assert payload['seasons'] == [2026, 2025, 2024]
    assert calls[1][1] == {'season': 2025, 'seasontype': 2}
    assert team['wins'] == 1 and team['losses'] == 0
    assert team['points_per_game'] == 90 and team['points_allowed_per_game'] == 80
    assert team['conference'] == 'Eastern'
    assert len(team['games']) == 1 and team['upcoming'][0]['points_for'] is None


def test_missing_scores_are_not_zero(monkeypatch):
    monkeypatch.setattr(content, '_fetch', lambda *a, **k: {'events': [event(score=None)]})
    team = content._team_schedule({'id': '20', 'abbreviation': 'ATL'}, 'wnba', 2026, [2026])
    assert team['wins'] is None and team['points_per_game'] is None
    assert not team['stats_available']


def test_provider_unavailable(monkeypatch):
    def unavailable(*a, **k):
        raise content.ProviderUnavailable()
    monkeypatch.setattr(content, '_fetch', unavailable)
    assert content.get_teams('nba')['teams'] == []
    assert 'unavailable' in content.get_highlights('nba')['message']


def test_clips_safe_video_and_external_fallback():
    clips = content._clips({'videos': [
        {'id': 123, 'headline': 'Real clip', 'links': {'source': {'href': 'https://espnmedia-cdn.akamaized.net/clip.mp4'}}},
        {'id': 124, 'links': {'source': {'href': 'javascript:alert(1)'}}},
        {'id': 'bad', 'links': {'web': {'href': 'https://espn.com.evil.test/clip'}}}]})
    assert len(clips) == 2
    assert clips[0]['video_url'].endswith('.mp4')
    assert clips[1]['video_url'] is None
    assert clips[1]['url'] == 'https://www.espn.com/video/clip/_/id/124'


def test_recent_completed_game_deduplicated_and_no_clips(monkeypatch):
    def fetch(league, path, **params):
        if path == 'scoreboard':
            return {'events': [{'id': '12', 'shortName': 'NY @ ATL', 'date': '2026-10-07T23:30Z',
                                'status': {'type': {'state': 'post', 'description': 'Final'}}}]}
        return {'videos': []}
    monkeypatch.setattr(content, '_fetch', fetch)
    result = content.get_highlights('wnba')
    assert len(result['games']) == 1
    assert result['games'][0]['clips'] == []
    assert 'not published' in result['message']


def test_team_and_opponent_aliases(monkeypatch):
    game = event()
    game['competitions'][0]['competitors'][1]['team']['abbreviation'] = 'NY'
    monkeypatch.setattr(content, '_fetch', lambda *a, **k: {'events': [game]})
    team = content._team_schedule({'id': '20', 'abbreviation': 'GS'}, 'wnba', 2026, [2026])
    assert team['abbrev'] == 'GSV'
    assert team['games'][0]['opponent'] == 'NYL'
    assert content.normalize_abbrev('nba', 'GS') == 'GSW'
    assert content.normalize_abbrev('wnba', 'POR') == 'PDX'


def test_highlight_matchup_aliases():
    event = {'competitions': [{'competitors': [
        {'homeAway': 'home', 'team': {'abbreviation': 'GS'}},
        {'homeAway': 'away', 'team': {'abbreviation': 'NY'}}]}]}
    assert content._matchup('nba', event) == 'NYK @ GSW'
    assert content._matchup('wnba', event) == 'NYL @ GSV'


def test_cup_championship_excluded_but_qualifiers_count(monkeypatch):
    qualifying = event('qualifying')
    qualifying['competitions'][0]['type'] = {'id': '1', 'slug': 'standard'}
    qualifying['competitions'][0]['notes'] = [{'headline': "WNBA Commissioner's Cup"}]
    championship = event('championship', score=130, opponent_score=120)
    championship['competitions'][0]['type'] = {'id': '39', 'slug': 'commissioners-cup'}
    championship['competitions'][0]['notes'] = [{'headline': "WNBA Commissioner's Cup Championship"}]
    monkeypatch.setattr(content, '_fetch', lambda *a, **k: {'events': [qualifying, championship]})
    team = content._team_schedule({'id': '20', 'abbreviation': 'ATL'}, 'wnba', 2026, [2026])
    assert team['wins'] == 1 and team['losses'] == 0
    assert team['points_per_game'] == 90
    assert [g['id'] for g in team['games']] == ['qualifying']


def test_response_cache_is_bounded(monkeypatch):
    class Reply:
        def raise_for_status(self): pass
        def json(self): return {'ok': True}
    monkeypatch.setattr(content.requests, 'get', lambda *a, **k: Reply())
    monkeypatch.setattr(content, '_CACHE_MAX', 5)
    content._cache.clear()
    for game in range(20):
        content._fetch_url('https://example.test/summary', event=str(game))
    assert len(content._cache) == 5
    content._cache.clear()
