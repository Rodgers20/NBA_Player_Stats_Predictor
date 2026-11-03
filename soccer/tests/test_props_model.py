import math

import pytest

from futball.props import model as m


def test_poisson_tails_match_closed_form():
    mu = 0.8
    assert m.at_least(mu, 1) == pytest.approx(1 - math.exp(-mu))
    assert m.at_least(mu, 2) == pytest.approx(1 - math.exp(-mu) * (1 + mu))


def test_negative_binomial_has_fatter_tail_but_same_mean_and_converges_to_poisson():
    mu = 0.3  # the regime player counts live in: overdispersion fattens the upper tail...
    assert m.at_least(mu, 2, phi=0.4) > m.at_least(mu, 2)
    assert m.at_least(mu, 1, phi=0.4) < m.at_least(mu, 1)  # ...while clumping lowers P(at least one)
    assert m.at_least(mu, 2, phi=1e-9) == pytest.approx(m.at_least(mu, 2), abs=1e-6)


def test_zero_rate_gives_zero_probability_and_probabilities_stay_in_range():
    assert m.at_least(0.0, 1) == 0.0
    assert 0.0 < m.at_least(5.0, 2, phi=0.3) < 1.0


def test_shrinkage_pulls_small_samples_to_the_prior_and_large_samples_to_the_data():
    prior = 0.20
    assert m.shrunk_rate(total=3, exposure=1.0, prior_rate=prior, strength=6) == pytest.approx((3 + 6 * prior) / 7)
    one_game = m.shrunk_rate(total=3, exposure=1.0, prior_rate=prior, strength=6)
    many = m.shrunk_rate(total=30, exposure=10.0, prior_rate=prior, strength=6)
    assert prior < one_game < 3.0 and many > one_game - 1e-9 and abs(many - 3.0) < abs(one_game - 3.0)


def test_exposure_counts_a_substitute_appearance_as_a_fraction_of_a_game():
    assert m.exposure(starts=6, subs=10) == pytest.approx(6 + 0.3 * 10)


def test_start_probability_blends_recent_and_season_and_decays_when_absent():
    steady = m.start_probability(starts=8, team_games=10, last5_starts=5, last5_games=5, gap=0)
    assert steady == pytest.approx(0.6 * 1.0 + 0.4 * 0.8)
    missing = m.start_probability(starts=8, team_games=10, last5_starts=2, last5_games=5, gap=4)
    assert missing < 0.25  # not seen for 4 team matches: treated as unavailable
    assert 0.0 <= m.start_probability(starts=0, team_games=0, last5_starts=0, last5_games=0, gap=0) <= 1.0


def test_expected_exposure_adds_substitute_chance_for_non_starters():
    assert m.expected_exposure(start_p=1.0, sub_rate=0.9) == pytest.approx(1.0)
    assert m.expected_exposure(start_p=0.0, sub_rate=0.5) == pytest.approx(0.5 * 0.3)


@pytest.mark.parametrize("games,market,expected", [
    (9, "anytime_goal", "high"), (5, "anytime_goal", "medium"), (3, "anytime_goal", "low"),
    (12, "shots_on_target", "high"),
    # Assists are the thinnest data: "low" below 8 games, and never better than "medium".
    (7, "assist", "low"), (8, "assist", "medium"), (20, "assist", "medium"),
])
def test_confidence_rules(games, market, expected):
    assert m.confidence(games, market) == expected


def test_gating_requires_two_games_and_a_quarter_chance_of_playing_and_not_a_keeper():
    assert m.eligible(games=2, start_p=0.25, position="F")
    assert not m.eligible(games=1, start_p=0.9, position="F")
    assert not m.eligible(games=9, start_p=0.2, position="M")
    assert not m.eligible(games=9, start_p=0.9, position="G")
