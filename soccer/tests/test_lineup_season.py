from unittest.mock import patch
from futball.lineup_season import aggregate


def game(event, goals, assists, rating=None, sub=False):
    stats = [{'name': 'totalGoals', 'value': goals}, {'name': 'goalAssists', 'value': assists}]
    if rating is not None:
        stats.append({'name': 'rating', 'value': rating})
    p = {'starter': not sub, 'subbedIn': sub, 'athlete': {'id':'yamal'}, 'stats':stats}
    others = [{'starter':True, 'athlete':{'id':str(i)}, 'stats':[]} for i in range(11 if sub else 10)]
    return {'header':{'id':event}, 'rosters':[{'team':{'displayName':'Barcelona'}, 'roster':[p,*others]}]}


def test_season_sums_match_contributions_including_substitutes():
    with patch('futball.match_lineups.canonical', side_effect=lambda x:x):
        result, covered = aggregate([game('1',2,1,8),game('2',4,2,10,sub=True)],'Barcelona',2)
    assert covered == 2
    assert result['yamal']['goals'] == 6
    assert result['yamal']['assists'] == 3
    assert result['yamal']['played'] == 2
    assert result['yamal']['rating'] == 9
    assert result['yamal']['rated_matches'] == 2


def test_missing_rating_excluded_and_missing_totals_not_zero():
    with patch('futball.match_lineups.canonical', side_effect=lambda x:x):
        result,_ = aggregate([game('1',2,None,9),game('2',0,1)],'Barcelona',2)
    assert result['yamal']['rating'] == 9
    assert result['yamal']['rated_matches'] == 1
    assert result['yamal']['assists'] is None


def test_incomplete_or_duplicate_matches_do_not_claim_season_totals():
    g=game('1',2,1)
    with patch('futball.match_lineups.canonical', side_effect=lambda x:x):
        result,covered=aggregate([g,g],'Barcelona',2)
    assert covered == 1
    assert result['yamal']['goals'] is None
    assert result['yamal']['rating'] is None
