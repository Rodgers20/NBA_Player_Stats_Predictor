"""Tests for utils.parlay_probability.

Regression coverage for the two defects found on 2026-09-10, where parlays
claiming 90-100% won only 44.4% (8/18):

  1. parlay_odds() recovered per-leg probability from vig-inflated American
     odds, so every leg was overstated by ~4% and anything above 0.945
     saturated at the 0.99 clamp in _prob_to_american().
  2. Per-leg probabilities came from a Normal CDF whose sigma was the
     player's last-10-game std, which excludes model prediction error.
"""

import math

import pytest

from utils.parlay_probability import (
    DEFAULT_RESID_SD,
    EMPIRICAL_RESID_SD,
    MAX_LEG_PROB,
    SAME_PLAYER_RHO,
    calibrate_leg_prob,
    devig_american,
    joint_probability,
)
from utils.parlay_builder import _prob_to_american


# ── de-vigging ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("true_prob", [0.55, 0.62, 0.70, 0.80, 0.85, 0.90])
def test_devig_inverts_prob_to_american(true_prob):
    """de-vig(prob_to_american(p)) must return p, not the vigged prob.

    The old code skipped the de-vig, inflating every leg by ~4%.
    """
    recovered = devig_american(_prob_to_american(true_prob))
    assert recovered == pytest.approx(true_prob, abs=0.01)


def test_devig_does_not_exceed_one():
    assert devig_american(-100000) <= 1.0


def test_raw_implied_is_higher_than_devigged():
    """Guards the specific bug: raw implied > true probability."""
    odds = _prob_to_american(0.80)
    raw_implied = abs(odds) / (abs(odds) + 100.0)
    assert raw_implied > devig_american(odds)


# ── leg calibration ───────────────────────────────────────────────────────────

def test_calibrate_uses_empirical_sigma_not_a_tight_one():
    """A 12.5-pt edge on PTS+AST+REB is ~90%, not 99%.

    Empirical residual SD for PTS+AST+REB is 9.55; the old path used a
    last-10 std near 5, which produced 0.99.
    """
    p = calibrate_leg_prob(0.99, stat="PTS+AST+REB", projection=30.0, line=17.5)
    expected = 1 - _norm_cdf((17.5 - 30.0) / EMPIRICAL_RESID_SD["PTS+AST+REB"])
    assert p == pytest.approx(expected, abs=0.01)
    assert p < 0.95


def test_calibrate_respects_empirical_ceiling():
    """No leg may claim more than the observed max hit rate (86% at the
    most extreme edge bucket, n=43). Ceiling is set to 0.90."""
    p = calibrate_leg_prob(0.999, stat="PTS", projection=200.0, line=1.0)
    assert p <= MAX_LEG_PROB


def test_calibrate_is_symmetric_for_under():
    over = calibrate_leg_prob(0.7, stat="PTS", projection=25.0, line=20.0,
                              direction="Over")
    under = calibrate_leg_prob(0.7, stat="PTS", projection=20.0, line=25.0,
                               direction="Under")
    assert over == pytest.approx(under, abs=1e-9)


def test_calibrate_falls_back_to_shrinkage_without_projection():
    """With no projection we cannot recompute, so shrink and clamp."""
    p = calibrate_leg_prob(0.99, stat="PTS")
    assert p <= MAX_LEG_PROB
    assert p < 0.99


def test_calibrate_unknown_stat_uses_default_sd():
    p = calibrate_leg_prob(0.8, stat="NOT_A_STAT", projection=10.0, line=5.0)
    expected = 1 - _norm_cdf((5.0 - 10.0) / DEFAULT_RESID_SD)
    assert p == pytest.approx(min(expected, MAX_LEG_PROB), abs=0.01)


def test_calibrate_is_monotonic_in_edge():
    """Bigger edge must never produce a lower probability (ranking preserved)."""
    probs = [
        calibrate_leg_prob(0.7, stat="PTS", projection=20.0 + g, line=20.0)
        for g in range(0, 12, 2)
    ]
    assert probs == sorted(probs)


# ── joint probability ─────────────────────────────────────────────────────────

def test_joint_equals_product_for_independent_players():
    """Cross-player rho measured at -0.002 (n=1436) -> treat as independent."""
    legs = [
        {"player": "A", "true_prob": 0.7},
        {"player": "B", "true_prob": 0.6},
        {"player": "C", "true_prob": 0.5},
    ]
    assert joint_probability(legs) == pytest.approx(0.7 * 0.6 * 0.5, abs=1e-4)


def test_joint_exceeds_product_for_same_player_legs():
    """Same-player rho = +0.196: positively correlated, so the joint must be
    ABOVE the independent product."""
    legs = [
        {"player": "A", "true_prob": 0.7},
        {"player": "A", "true_prob": 0.6},
    ]
    j = joint_probability(legs)
    assert j > 0.7 * 0.6
    assert j < 0.6  # cannot exceed the weakest marginal


def test_joint_never_exceeds_weakest_leg():
    legs = [{"player": f"P{i}", "true_prob": p}
            for i, p in enumerate([0.9, 0.85, 0.4])]
    assert joint_probability(legs) <= 0.4


def test_joint_of_empty_and_single():
    assert joint_probability([]) == 0.0
    assert joint_probability([{"player": "A", "true_prob": 0.63}]) == pytest.approx(0.63)


def test_joint_decreases_with_more_legs():
    base = [{"player": f"P{i}", "true_prob": 0.75} for i in range(3)]
    more = base + [{"player": "PX", "true_prob": 0.75}]
    assert joint_probability(more) < joint_probability(base)


def test_zero_rho_reduces_to_product():
    """The one-factor copula must collapse to the independent product."""
    legs = [{"player": "A", "true_prob": 0.8}, {"player": "A", "true_prob": 0.7}]
    assert joint_probability(legs, same_player_rho=0.0) == pytest.approx(0.8 * 0.7,
                                                                        abs=1e-4)


# ── the regression that started this ──────────────────────────────────────────

def test_three_legs_of_true_095_is_not_97_percent():
    """The exact shape of the 18 bad parlays: 3 legs, claimed 97.0%, won 44%.

    Independent product of 0.95 is 85.7%. The old code returned 97.0%
    because it round-tripped through vigged odds and hit the 0.99 clamp.
    """
    legs = [{"player": f"P{i}", "true_prob": 0.95} for i in range(3)]
    assert joint_probability(legs) == pytest.approx(0.95 ** 3, abs=1e-3)
    assert joint_probability(legs) < 0.90


def test_calibrated_pipeline_cannot_reach_97_percent_on_three_legs():
    """End-to-end guard: even maximally confident legs cannot claim 97%."""
    legs = [
        {"player": f"P{i}",
         "true_prob": calibrate_leg_prob(0.99, stat="PTS+AST+REB",
                                         projection=99.0, line=1.0)}
        for i in range(3)
    ]
    assert joint_probability(legs) < 0.75


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
