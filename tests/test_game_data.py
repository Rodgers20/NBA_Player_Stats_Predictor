from pathlib import Path
import pandas as pd
import pytest
from utils.game_data import validate_team_games, write_history, load_cached_nba


def pair():
    return pd.DataFrame([{'GAME_ID':'1','GAME_DATE':'2026-01-01','TEAM_ABBREVIATION':'ATL','PTS':110,'is_home':1},
                         {'GAME_ID':'1','GAME_DATE':'2026-01-01','TEAM_ABBREVIATION':'BOS','PTS':100,'is_home':0}])


def test_pairs_reject_incomplete_and_derive_actual_opponent():
    d=pair();d.loc[2]=['2','2026-01-02','ATL',90,1]
    result=validate_team_games(d)
    assert len(result)==2
    assert result.OPP_PTS.tolist()==[100,110]
    assert result.PLUS_MINUS.tolist()==[10,-10]


def test_invalid_home_and_missing_scores_are_rejected():
    d=pair();d['is_home']=1
    assert validate_team_games(d).empty
    d=pair();d.loc[0,'PTS']=None
    assert validate_team_games(d).empty


def test_empty_refresh_preserves_last_good_file(tmp_path):
    output=tmp_path/'history.csv';output.write_text('last good')
    with pytest.raises(ValueError):write_history(pair().iloc[:1],output)
    assert output.read_text()=='last good'


def test_metadata_records_coverage_not_fake_freshness(tmp_path):
    report=write_history(pair(),tmp_path/'history.csv')
    assert report['games']==1
    assert report['last_game']=='2026-01-01'
    assert report['historical_pregame_injuries'] is False


def test_kaggle_direct_targets_and_season_types(tmp_path):
    common={'gameDateTimeEst':'2026-02-01 19:00:00','gameType':'Regular Season','numMinutes':240}
    rows=[{**common,'gameId':22500001,'teamId':1610612737,'opponentTeamId':1610612738,'home':1,'teamScore':101},
          {**common,'gameId':22500001,'teamId':1610612738,'opponentTeamId':1610612737,'home':0,'teamScore':99}]
    from utils.game_data import STAT_MAP
    for row in rows:
        for col in STAT_MAP:row.setdefault(col,1)
    pd.DataFrame(rows).to_csv(tmp_path/'TeamStatistics.csv',index=False)
    pd.DataFrame([{'gameId':22500001,'arenaName':'Arena','arenaCity':'Atlanta','arenaState':'GA'}]).to_csv(tmp_path/'Games.csv',index=False)
    result=load_cached_nba(cache_path=tmp_path)
    assert result.PTS.tolist()==[101,99]
    assert set(result.SEASON)=={'2025-26'}
    assert set(result.arenaName)=={'Arena'}


def test_cached_supplement_has_regular_season_id(tmp_path):
    """Supplement rows must survive canonical SEASON_ID eligibility filtering."""
    from unittest.mock import patch
    raw=pair()
    raw['SEASON_ID']=['22025','22025']
    raw['SEASON_TYPE']='Regular Season'
    with patch('utils.game_features.normalize_team_games',return_value='canonical') as normalize:
        with patch('utils.game_data.pd.read_csv',return_value=raw):
            from utils.game_data import load_game_team_history
            assert load_game_team_history('nba')=='canonical'
            normalize.assert_called_once_with(raw,'nba')
