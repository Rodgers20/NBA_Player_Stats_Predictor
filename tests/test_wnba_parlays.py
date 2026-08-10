"""Tests for utils.wnba_parlays."""

import pytest

from utils.wnba_parlays import build_wnba_parlays, _decimal_to_american
from utils.wnba_props import WnbaProp


def _prop(name, stat="PTS", pick="OVER", ev=0.15, hit_prob=0.65,
          over_price=-110, under_price=-110, confidence="MED"):
    return WnbaProp(
        player_name=name, team="LVA", stat=stat, line=15.5, projected=17.0,
        edge=1.5, pick=pick, hit_prob=hit_prob, ev=ev,
        over_price=over_price, under_price=under_price, bookmaker="FanDuel",
        confidence=confidence,
    )


def test_decimal_to_american_positive_and_negative():
    assert _decimal_to_american(2.0) == 100
    assert _decimal_to_american(3.0) == 200
    assert _decimal_to_american(1.5) == -200


def test_build_parlays_produces_multiple_sizes():
    # 6 healthy props → should give 2/3/4-leg parlays
    props = [_prop(f"Player {i}") for i in range(6)]
    parlays = build_wnba_parlays(props, leg_counts=(2, 3, 4), max_per_size=3)
    sizes = {len(p.legs) for p in parlays}
    assert sizes == {2, 3, 4}


def test_no_two_legs_from_same_player():
    props = [
        _prop("Duplicate Player", stat="PTS"),
        _prop("Duplicate Player", stat="AST"),  # same player, different stat
        _prop("Other Player 1"),
        _prop("Other Player 2"),
    ]
    parlays = build_wnba_parlays(props, leg_counts=(2,), max_per_size=10)
    for p in parlays:
        names = [leg.player_name for leg in p.legs]
        assert len(set(names)) == len(names), f"repeated player in parlay: {names}"


def test_filters_low_ev_and_low_hit_prob():
    # Only 3 legs pass filter; the low-EV / low-hit props should be excluded
    props = [
        _prop("Good 1", ev=0.20, hit_prob=0.7),
        _prop("Good 2", ev=0.18, hit_prob=0.68),
        _prop("Good 3", ev=0.15, hit_prob=0.60),
        _prop("Bad EV",   ev=0.01, hit_prob=0.7),
        _prop("Bad Prob", ev=0.20, hit_prob=0.30),
    ]
    parlays = build_wnba_parlays(props, leg_counts=(2,), min_leg_ev=0.05, min_leg_hit_prob=0.55)
    for p in parlays:
        for leg in p.legs:
            assert leg.player_name in {"Good 1", "Good 2", "Good 3"}


def test_parlay_ev_is_correctly_combined():
    # Two +100 legs each at 60% hit → combined decimal 4.0, combined hit 0.36
    # EV = 0.36 * (4 - 1) - 0.64 = 1.08 - 0.64 = 0.44
    props = [
        _prop("A", ev=0.20, hit_prob=0.60, over_price=100),
        _prop("B", ev=0.20, hit_prob=0.60, over_price=100),
    ]
    parlays = build_wnba_parlays(props, leg_counts=(2,), max_per_size=1,
                                 min_leg_ev=0.05, min_leg_hit_prob=0.55)
    assert len(parlays) == 1
    p = parlays[0]
    assert p.combined_decimal == pytest.approx(4.0, abs=0.01)
    assert p.hit_prob == pytest.approx(0.36, abs=0.001)
    assert p.ev == pytest.approx(0.44, abs=0.01)


def test_returns_empty_when_no_positive_ev_combos():
    # All legs very low EV — filter rules them out entirely
    props = [_prop(f"Weak {i}", ev=0.001, hit_prob=0.51) for i in range(4)]
    assert build_wnba_parlays(props, leg_counts=(2,)) == []
