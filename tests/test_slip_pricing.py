import math

import pytest
from fastapi.testclient import TestClient

from api.main import app
from utils.parlay_probability import hit_correlation, joint_probability_sgp, stat_correlation

client = TestClient(app)


def leg(player='A', team='LAL', stat='PTS', side='Over', prob=0.55, game='g1', **extra):
    return dict(player=player, team=team, stat=stat, side=side, prob=prob, game=game, **extra)


def test_uncorrelated_legs_multiply_exactly():
    legs = [leg(game='g1'), leg(player='B', game='g2')]
    assert joint_probability_sgp(legs) == pytest.approx(0.55 * 0.55)
    assert stat_correlation(*legs) == (0.0, '')


def test_pra_component_is_strongly_positive_and_beats_independence():
    legs = [leg(stat='PTS'), leg(stat='PTS+REB+AST')]
    assert stat_correlation(*legs)[0] == 0.60
    assert joint_probability_sgp(legs) > 0.55 * 0.55 + 0.05


def test_teammate_points_mildly_negative_and_pts_ast_mildly_positive():
    pts_pts = [leg(), leg(player='B')]
    assert stat_correlation(*pts_pts)[0] == -0.05
    assert joint_probability_sgp(pts_pts) < 0.55 * 0.55
    pts_ast = [leg(), leg(player='B', stat='AST')]
    assert stat_correlation(*pts_ast)[0] == 0.10
    assert joint_probability_sgp(pts_ast) > 0.55 * 0.55


def test_under_flips_the_sign_of_the_hit_correlation():
    assert hit_correlation(leg(), leg(stat='PTS+REB', side='Under'))[0] == -0.60
    assert hit_correlation(leg(side='Under'), leg(stat='PTS+REB', side='Under'))[0] == 0.60


def test_opponents_share_pace_and_missing_team_is_independent():
    assert stat_correlation(leg(), leg(player='B', team='BOS'))[0] == 0.04
    assert stat_correlation(leg(), leg(player='B', team=''))[0] == 0.0


def test_result_is_deterministic_and_never_exceeds_weakest_leg():
    legs = [leg(stat='PTS', prob=0.3), leg(stat='PTS+REB+AST', prob=0.9)]
    assert joint_probability_sgp(legs) == joint_probability_sgp(legs) <= 0.3


def test_route_shape_and_notes():
    body = {'legs': [leg(price=-110), leg(stat='PTS+REB+AST', price=-115)]}
    data = client.post('/api/slip/price', json=body).json()
    assert data['independent_prob'] == pytest.approx(0.55 * 0.55)
    assert data['correlated_prob'] > data['independent_prob']
    assert data['fair_odds'] > 0 and isinstance(data['fair_odds'], int)
    assert data['per_leg_corr'][0] == {'leg': 1, 'notes': [{'with': 2, 'rho': 0.6, 'reason': 'same player, one prop is part of the other'}]}
    assert math.isclose(data['per_leg_corr'][1]['notes'][0]['rho'], 0.6)


def test_route_without_correlation_has_no_notes():
    data = client.post('/api/slip/price', json={'legs': [leg(game='a'), leg(player='B', game='b')]}).json()
    assert data['correlated_prob'] == data['independent_prob']
    assert all(item['notes'] == [] for item in data['per_leg_corr'])


@pytest.mark.parametrize('legs', [
    [], [leg()] * 13, [leg(prob=0)], [leg(prob=1)], [leg(side='Push')],
    [leg(stat='DUNKS')], [leg(price=50)], [leg(player='')],
])
def test_route_rejects_invalid_input(legs):
    response = client.post('/api/slip/price', json={'legs': legs})
    assert response.status_code == 422
