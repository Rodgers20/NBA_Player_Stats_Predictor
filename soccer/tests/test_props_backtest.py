import math

import pytest

from futball.props import backtest as b


def test_log_loss_of_a_perfect_confident_and_a_coin_flip():
    assert b.log_loss([(1.0, 1), (0.0, 0)]) == pytest.approx(0.0, abs=2e-4)  # probabilities are clipped at 1e-4
    assert b.log_loss([(0.5, 1), (0.5, 0)]) == pytest.approx(math.log(2))


def test_baseline_predicts_the_observed_rate_everywhere():
    pairs = [(0.9, 1), (0.1, 0), (0.1, 0), (0.1, 0)]
    base = b.baseline_log_loss(pairs)
    rate = 0.25
    assert base == pytest.approx(-(0.25 * math.log(rate) + 0.75 * math.log(1 - rate)))


def test_calibration_bins_report_count_mean_prediction_and_hit_rate():
    pairs = [(0.04, 0), (0.06, 0), (0.07, 1), (0.55, 1), (0.65, 0)]
    table = {row["bin"]: row for row in b.calibration(pairs, edges=(0.0, 0.1, 0.5, 1.0))}
    assert table["0-10%"]["n"] == 3 and table["0-10%"]["hit_rate"] == pytest.approx(1 / 3)
    assert table["0-10%"]["mean_pred"] == pytest.approx((0.04 + 0.06 + 0.07) / 3)
    assert table["50-100%"]["n"] == 2 and table["10-50%"]["n"] == 0


def test_outcome_counts_a_missing_player_as_a_miss_and_thresholds_by_market():
    stats = {"goals": 1, "assists": 0, "sot": 2}
    assert b.outcome("anytime_goal", None, stats) == 1
    assert b.outcome("assist", None, stats) == 0
    assert b.outcome("shots_on_target", 1.5, stats) == 1
    assert b.outcome("shots_on_target", 1.5, {"goals": 0, "assists": 0, "sot": 1}) == 0
    assert b.outcome("anytime_goal", None, None) == 0  # did not play


def test_verdict_is_honest_when_the_model_does_not_beat_the_baseline():
    assert "does not beat" in b.verdict(model_ll=0.30, base_ll=0.29, n=500)
    assert "beats" in b.verdict(model_ll=0.27, base_ll=0.29, n=500)
    assert "too few" in b.verdict(model_ll=0.27, base_ll=0.29, n=20)
