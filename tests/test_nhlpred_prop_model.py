import numpy as np
import pandas as pd

from nhlpred import prop_model as pm


def test_line_probabilities_are_monotone_and_dispersion_detects_overdispersion():
    mu = np.array([0.3, 1.0, 3.0])
    p = pm.probs(mu, 0.2, pm.MARKETS["sog"]["lines"])
    lines = sorted(p)
    for lo, hi in zip(lines, lines[1:]):
        assert (p[lo] >= p[hi]).all()
    rng = np.random.default_rng(0)
    mu_big = np.full(20000, 3.0)
    y = rng.negative_binomial(5, 5 / (5 + 3.0), 20000)       # var = mu + mu^2/5 -> alpha 0.2
    assert 0.12 < pm.dispersion(y, mu_big) < 0.28
    assert pm.dispersion(rng.poisson(3.0, 20000), mu_big) < 0.05


def test_calibration_window_is_the_last_game_days_before_the_cut():
    days = pd.date_range("2026-01-01", periods=100)
    done = pd.DataFrame({"date": days})
    start = pm.cal_window_start(done, pd.Timestamp("2026-04-01"))
    assert start == days[-pm.CAL_GAME_DAYS] or start == days[days < pd.Timestamp("2026-04-01")][-pm.CAL_GAME_DAYS]


def test_confidence_tiers():
    assert pm.confidence(35) == "high" and pm.confidence(20) == "medium" and pm.confidence(4) == "low"
