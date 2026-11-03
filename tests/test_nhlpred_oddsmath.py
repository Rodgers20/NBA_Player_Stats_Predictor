import numpy as np

from nhlpred.oddsmath import american_to_prob, fair_american, line_to_k, p_at_least


def test_fair_american_roundtrip():
    assert fair_american(0.6) == -150 and fair_american(0.4) == 150 and fair_american(0.5) == -100
    assert fair_american(0) is None and fair_american(None) is None
    for p in (0.1, 0.35, 0.5, 0.77):
        assert abs(american_to_prob(fair_american(p)) - p) < 0.01


def test_tails():
    assert abs(float(p_at_least(1.0, 1)) - (1 - np.exp(-1))) < 1e-9
    assert line_to_k(2.5) == 3 and line_to_k(0.5) == 1
    # overdispersion fattens the upper tail, keeps the mean
    assert float(p_at_least(3.0, 8, alpha=0.3)) > float(p_at_least(3.0, 8))
    assert float(p_at_least(2.0, 0)) == 1.0
