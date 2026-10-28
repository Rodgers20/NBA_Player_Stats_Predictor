import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from api.routes import games as games_route
from utils import matchup


def log(day, team, opp, home, pts, opp_pts, season='2025-26', kind='Regular Season'):
    return dict(GAME_DATE=day, TEAM_ABBREVIATION=team, OPP_TEAM_ABBREVIATION=opp, is_home=int(home), PTS=pts, OPP_PTS=opp_pts,
                WL='W' if pts > opp_pts else 'L', SEASON=season, SEASON_TYPE=kind)


def both_sides(day, home, away, home_pts, away_pts, **kw):
    return [log(day, home, away, True, home_pts, away_pts, **kw), log(day, away, home, False, away_pts, home_pts, **kw)]


HISTORY = pd.DataFrame(
    both_sides('2026-01-05', 'ATL', 'ORL', 120, 100) + both_sides('2026-02-10', 'ORL', 'ATL', 105, 110)
    + both_sides('2026-03-01', 'ATL', 'ORL', 99, 101) + both_sides('2026-03-05', 'ATL', 'BOS', 90, 130)
    + both_sides('2026-03-09', 'BOS', 'ORL', 100, 90))


def test_head_to_head_is_newest_first_with_a_record():
    h2h = matchup.head_to_head(HISTORY, 'ATL', 'ORL')
    assert [g['date'] for g in h2h['games']] == ['2026-03-01', '2026-02-10', '2026-01-05']
    assert h2h['games'][0] == dict(date='2026-03-01', home='ATL', away='ORL', home_pts=99, away_pts=101, winner='ORL', kind='Regular Season')
    assert h2h['record'] == {'ATL': 2, 'ORL': 1}
    assert matchup.head_to_head(HISTORY, 'ATL', 'XXX') == dict(games=[], record={'ATL': 0, 'XXX': 0})


def test_home_and_road_form_use_only_that_venue():
    home = matchup.venue_form(HISTORY, 'ATL', at_home=True)
    assert (home['games'], home['wins'], home['losses']) == (3, 1, 2)
    assert home['pts_for'] == round((120 + 99 + 90) / 3, 1) and home['margin'] == round((20 - 2 - 40) / 3, 1)
    road = matchup.venue_form(HISTORY, 'ATL', at_home=False)
    assert (road['games'], road['wins']) == (1, 1) and road['pts_for'] == 110
    assert matchup.venue_form(HISTORY, 'XXX', at_home=True) is None


STATS = pd.DataFrame({'TEAM_ABBREVIATION': list('ABCDEF'), 'SEASON': '2025-26',
                      'PTS': [120, 118, 116, 110, 108, 100], 'TOV': [11, 12, 13, 14, 15, 16]})
DEFENSE = pd.DataFrame({'TEAM_ABBREVIATION': list('ABCDEF'), 'SEASON': '2025-26', 'OPP_PTS': [100, 104, 108, 112, 116, 120]})


def test_profile_ranks_every_metric_and_names_strengths_and_weaknesses():
    profile = matchup.team_profile(STATS, DEFENSE, 'A')
    by_label = {m['label']: m for m in profile['metrics']}
    assert by_label['Scoring']['rank'] == 1 and by_label['Ball security']['rank'] == 1  # fewer turnovers is better
    assert by_label['Scoring defense']['rank'] == 1 and by_label['Scoring']['of'] == 6
    assert {m['label'] for m in profile['strengths']} == {'Scoring', 'Ball security', 'Scoring defense'}
    assert profile['weaknesses'] == []
    worst = matchup.team_profile(STATS, DEFENSE, 'F')
    assert {m['label'] for m in worst['weaknesses']} == {'Scoring', 'Ball security', 'Scoring defense'}
    assert matchup.team_profile(STATS, DEFENSE, 'ZZZ') is None


def test_edges_flag_a_strong_attack_against_a_weak_defense():
    edges = matchup.edges(STATS, DEFENSE, attacker='A', defender='F')
    assert edges and edges[0]['team'] == 'A' and edges[0]['label'] == 'Scoring' and edges[0]['gap'] == 5
    assert matchup.edges(STATS, DEFENSE, attacker='F', defender='A') == []


def test_route_returns_all_sections_and_rejects_bad_abbreviations(monkeypatch):
    monkeypatch.setattr(games_route, 'matchup_frames', lambda league: (HISTORY, STATS, DEFENSE))
    client = TestClient(app)
    body = client.get('/api/games/matchup?league=nba&home=ATL&away=ORL').json()
    assert body['h2h']['record'] == {'ATL': 2, 'ORL': 1}
    assert body['splits']['home']['team'] == 'ATL' and body['splits']['away']['team'] == 'ORL'
    assert set(body) >= {'h2h', 'splits', 'profiles', 'edges', 'season'}
    assert client.get('/api/games/matchup?league=nba&home=atl!&away=ORL').status_code == 422
