import numpy as np
import pytest

from utils.market_evaluation import (MAX_CREDIBLE_EV, MIN_SAMPLE, credibility_problem, evaluate_market,
                                     no_vig_probability, shrink_weight)

RESIDUALS = np.zeros(40)   # the model has never been wrong, so the raw probability is extreme


def test_shrink_weight_grows_with_sample_and_is_half_at_ten_games():
    assert shrink_weight(0) == 0
    assert shrink_weight(10) == pytest.approx(0.5)
    assert shrink_weight(3) < shrink_weight(10) < shrink_weight(30) < 1


def test_no_vig_probability_removes_the_book_margin():
    assert no_vig_probability(-110, -110) == pytest.approx(0.5)
    assert no_vig_probability(-150, 130) == pytest.approx(0.5798, abs=1e-3)   # 0.600 / (0.600 + 0.4348)
    assert no_vig_probability(-150, 130) + no_vig_probability(130, -150) == pytest.approx(1)


def test_unshrunk_evaluation_is_unchanged_without_both_prices_and_a_sample():
    plain = evaluate_market(12, 10.5, -110, RESIDUALS, 'over')
    assert 'raw_model_prob' not in plain
    assert evaluate_market(12, 10.5, -110, RESIDUALS, 'over', other_price=-110) == plain   # no sample: no shrink


def test_probability_is_pulled_toward_the_market_by_less_data():
    raw = evaluate_market(12, 10.5, -110, RESIDUALS, 'over')
    thin = evaluate_market(12, 10.5, -110, RESIDUALS, 'over', other_price=-110, sample=3)
    full = evaluate_market(12, 10.5, -110, RESIDUALS, 'over', other_price=-110, sample=30)
    assert thin['market_prob'] == pytest.approx(0.5)
    assert 0.5 < thin['model_prob'] < full['model_prob'] < raw['model_prob']
    assert thin['ev'] < full['ev'] < raw['ev']
    assert thin['raw_model_prob'] == pytest.approx(raw['model_prob'])
    assert thin['shrink_weight'] == pytest.approx(shrink_weight(3))


def test_pushes_stay_out_of_the_shrunk_win_probability():
    # a whole-number line the model lands on often: win + push + loss must still sum to one
    result = evaluate_market(10, 10, -110, np.array([0] * 20 + [1] * 10 + [-1] * 10), 'over', other_price=-110, sample=10)
    assert result['model_prob'] + result['push_prob'] <= 1


def test_credibility_hides_thin_samples_and_implausible_edges():
    assert 'Thin sample' in credibility_problem(MIN_SAMPLE - 1, 0.05)
    assert 'Thin sample' in credibility_problem(None, 0.05)
    assert 'not credible' in credibility_problem(10, MAX_CREDIBLE_EV + 0.01)
    assert credibility_problem(10, MAX_CREDIBLE_EV) is None
    assert credibility_problem(MIN_SAMPLE, 0.08) is None
