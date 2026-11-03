from unittest.mock import patch
from futball.season_ratings import apply_ratings


def data():
    return {'details':{'name':'Barcelona','primaryLeagueId':87}, 'overview':{'season':'2026/2027'}, 'squad':{'squad':[{'title':'attackers','members':[{'id':1,'name':'Lamine Yamal','shirtNumber':10,'rating':8.48}]}]}}


def test_verified_season_average_keeps_goals_and_does_not_invent_match_count():
    p={'name':'Lamine Yamal','shirt':'10','goals':6,'rating':None}
    with patch('futball.match_lineups.canonical',side_effect=lambda x:x):
        assert apply_ratings(data(),[p],'Barcelona','SP1','2026/2027') == 1
    assert p['rating'] == 8.48
    assert p['goals'] == 6
    assert p['rating_source'] == 'FotMob'
    assert p['rated_matches'] is None


def test_wrong_season_competition_club_and_shirt_are_rejected():
    with patch('futball.match_lineups.canonical',side_effect=lambda x:x):
        for team,code,season,shirt in [('Barcelona','SP1','2025/2026','10'),('Barcelona','E0','2026/2027','10'),('Betis','SP1','2026/2027','10'),('Barcelona','SP1','2026/2027','11')]:
            assert apply_ratings(data(),[{'name':'Lamine Yamal','shirt':shirt}],team,code,season) == 0


def test_ambiguous_identity_and_invalid_rating_rejected():
    d=data(); member=d['squad']['squad'][0]['members'][0]
    with patch('futball.match_lineups.canonical',side_effect=lambda x:x):
        for value in [None,0,11,float('nan'),True]:
            member['rating']=value
            assert apply_ratings(d,[{'name':'Lamine Yamal','shirt':'10'}],'Barcelona','SP1','2026/2027') == 0
        member['rating']=8
        d['squad']['squad'][0]['members'].append(dict(member))
        assert apply_ratings(d,[{'name':'Lamine Yamal','shirt':'10'}],'Barcelona','SP1','2026/2027') == 0
