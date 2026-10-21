from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from api.main import app
from utils import defense_rank

client = TestClient(app)


@pytest.fixture
def league_data(tmp_path, monkeypatch):
    """Five-team WNBA-style files (rank 1 in the source files = toughest defense)."""
    teams = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
    lines = ['TEAM_ABBREVIATION,SEASON,POSITION,GP,PTS,AST,REB,FG3M,PTS_RANK,AST_RANK,REB_RANK,FG3M_RANK']
    for i, team in enumerate(teams):
        lines.append(f'{team},2025,G,10,{10 + i},3,4,1,1,1,1,1')       # older season, ignored
        lines.append(f'{team},2026,G,10,{20 + i},{3 + i},{4 + i},1,1,1,1,1')
    (tmp_path / 'defense_vs_position.csv').write_text('\n'.join(lines))
    (tmp_path / 'player_positions.csv').write_text(
        'PLAYER_ID,PLAYER_NAME,TEAM_ABBREVIATION,POSITION,SEASON\n1,Guard One,AAA,G,2026\n2,Wing Two,BBB,F,2026\n')
    monkeypatch.setattr(defense_rank, 'get_config', lambda league: SimpleNamespace(data_dir=tmp_path))
    defense_rank._load.cache_clear()
    yield
    defense_rank._load.cache_clear()


def test_rank_one_is_softest_and_grade_follows(league_data):
    soft = defense_rank.matchup('wnba', 'Guard One', 'EEE', 'PTS')      # allows 24, the most
    tough = defense_rank.matchup('wnba', 'Guard One', 'AAA', 'PTS')
    assert soft == dict(def_rank=1, def_rank_of=5, matchup_grade='A')
    assert tough == dict(def_rank=5, def_rank_of=5, matchup_grade='F')


def test_unknown_inputs_give_empty_matchup(league_data):
    none = defense_rank.NO_MATCHUP
    assert defense_rank.matchup('wnba', 'Nobody', 'EEE', 'PTS') == none
    assert defense_rank.matchup('wnba', 'Wing Two', 'EEE', 'PTS') == none    # no F rows
    assert defense_rank.matchup('wnba', 'Guard One', 'ZZZ', 'PTS') == none
    assert defense_rank.matchup('wnba', 'Guard One', 'EEE', 'STL') == none
    assert defense_rank.matchup('wnba', 'Guard One', '', 'PTS') == none


def test_grade_bands():
    assert [defense_rank.grade(r, 30) for r in (1, 6, 7, 12, 13, 18, 19, 24, 25, 30)] == list('AABBCCDDFF')


def test_endpoint_returns_latest_season_rankings(league_data):
    body = client.get('/api/defense?league=wnba&stat=PTS').json()
    assert body['season'] == '2026' and body['stat'] == 'PTS' and body['league'] == 'wnba'
    assert len(body['rankings']) == 5
    assert body['rankings'][0] == dict(team='EEE', position='G', stat='PTS', allowed=24.0, rank=1, of=5)
    assert len(client.get('/api/defense?league=wnba').json()['rankings']) == 15
    assert client.get('/api/defense?league=wnba&stat=STL').status_code == 422


def test_missing_files_return_empty_not_error(tmp_path, monkeypatch):
    monkeypatch.setattr(defense_rank, 'get_config', lambda league: SimpleNamespace(data_dir=tmp_path))
    body = client.get('/api/defense?league=nba').json()
    assert body['rankings'] == [] and body['season'] is None
    assert defense_rank.matchup('nba', 'x', 'BOS', 'PTS') == defense_rank.NO_MATCHUP


def test_real_nba_files_load_if_present():
    body = defense_rank.rankings('nba', 'PTS')
    assert body['season'] is None or all(1 <= r['rank'] <= r['of'] for r in body['rankings'])
