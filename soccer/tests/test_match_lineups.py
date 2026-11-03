from unittest.mock import patch

from futball import match_lineups as m


def member(pid, position='F', starter=True):
    return {'starter': starter, 'athlete': {'id': str(pid), 'displayName': f'Player {pid}'},
            'position': {'abbreviation': position, 'displayName': position},
            'stats': [{'name': k, 'value': 0} for k in ('totalGoals', 'goalAssists', 'shotsOnTarget', 'yellowCards', 'redCards', 'ownGoals', 'saves')]}


def test_only_confirmed_starters_and_provider_ids_not_public_ids():
    rows = m.starters({'roster': [member(1), member(1), member(2, starter=False), member(3, starter=None)]})
    assert len(rows) == 1
    assert rows[0]['provider_id'] == '1'
    assert rows[0]['player_id'] is None
    assert rows[0]['rating'] is None


def test_unknown_position_is_not_a_forward():
    p = m.starters({'roster': [member(1, 'unknown')]})
    assert p[0]['group'] is None
    assert m.pitch_slots(p)[0]['abbreviation'] == '?'


def test_missing_stats_not_zero_or_rating():
    p = m.starters({'roster': [{'starter': True, 'athlete': {'id': '1', 'displayName': 'A'}}]})[0]
    assert p['goals'] is None
    assert m.contribution_score(p, 0) is None


def test_match_score_rewards_real_contributions_and_caps():
    p = m.starters({'roster': [member(1)]})[0]
    base = m.contribution_score(p, 0)
    p['stats']['totalGoals'] = 2
    assert m.contribution_score(p, 0) > base
    p['stats']['totalGoals'] = 100
    assert m.contribution_score(p, 0) == 10
    p['stats']['totalGoals'] = 0
    p['stats']['redCards'] = 1
    assert m.contribution_score(p, 0) < base


def fixture(day=4, status='FINISHED', home='Home', away='Away'):
    return {'id': 1, 'matchday': day, 'status': status, 'utcDate': '2026-09-01T19:00:00Z',
            'homeTeam': {'name': home}, 'awayTeam': {'name': away}, 'score': {'fullTime': {'home': 1, 'away': 0}}}


def test_match_join_requires_both_sides_date_and_finished():
    event = {'id': '123', 'date': '2026-09-01T19:00Z', 'status': {'type': {'completed': True}},
             'competitions': [{'competitors': [{'homeAway': 'home', 'team': {'displayName': 'Home'}}, {'homeAway': 'away', 'team': {'displayName': 'Different'}}]}]}
    with patch.object(m, 'canonical', side_effect=lambda x:x), patch.object(m, 'fetch', return_value={'events':[event]}) as get:
        assert m.match_summary(fixture(), 'SP1') == {}
        # ESPN 400s on date ranges: the day before, the day and the day after are three single-date calls.
        assert get.call_count == 3
        assert [c.args[2]['dates'] for c in get.call_args_list] == ['20260831', '20260901', '20260902']


def test_latest_does_not_fall_back_to_an_older_full_lineup():
    old = fixture()
    new = {**old, 'utcDate': '2026-09-02T19:00:00Z'}
    with patch.object(m, 'canonical', side_effect=lambda x:x), patch.object(m, 'fixtures', side_effect=lambda c: [old,new] if c == 'SP1' else []), patch.object(m, 'match_summary', return_value={}) as summary:
        result = m.latest('Home', 'SP1')
    assert result['filled'] == 0
    assert result['match_date'] == new['utcDate']
    assert summary.call_args.args[0] == new


def test_team_of_week_prefers_complete_round_and_keeps_roles():
    lineup = [member(i, pos) for i,pos in enumerate(['G']+['D']*4+['M']*3+['F']*3)]
    body = {'rosters': [{'homeAway': side, 'team': {'displayName': side}, 'roster': lineup} for side in ['home','away']]}
    with patch.object(m, 'canonical', side_effect=lambda x:x), patch.object(m, 'fixtures', return_value=[fixture(4),fixture(5),fixture(5,'SCHEDULED')]), patch.object(m, 'match_summary', return_value=body):
        result = m.team_of_week('SP1')
    assert result['match_week'] == 4
    players = [s['player'] for s in result['players']]
    assert len(players) == 11
    assert len({p['provider_id'] for p in players}) == 11
    assert sum(p['group'] == 'GK' for p in players) == 1
    assert all(p['player_id'] is None for p in players)


def test_cache_failure_marked_stale(tmp_path):
    with patch.object(m, 'RAW', tmp_path), patch.object(m._cache, 'is_fresh', return_value=False), patch.object(m._cache, 'read_json', return_value={'rosters': []}), patch.object(m.httpx, 'get', side_effect=m.httpx.ConnectError('offline')):
        assert m.fetch('SP1','summary',{'event':'1'})['_stale'] is True
