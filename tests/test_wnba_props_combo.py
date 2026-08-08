"""Tests for combo props (PTS+REB, PTS+AST, PTS+REB+AST) in wnba_props."""

import pandas as pd
import pytest

from utils.wnba_props import generate_wnba_props


class _FakeModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, feat_row):
        return {"predicted_value": self.value}


def _getter(models: dict):
    def _fn(stat):
        return models.get(stat)
    return _fn


def _player_df(rows: int, pts: float, ast: float, reb: float):
    return pd.DataFrame({
        "PLAYER_NAME": ["Star"] * rows,
        "TEAM_ABBREVIATION": ["LVA"] * rows,
        "_date": pd.date_range("2026-07-01", periods=rows),
        "PTS": [pts] * rows, "AST": [ast] * rows, "REB": [reb] * rows,
        "FG3M": [2] * rows,
    })


def test_pts_plus_reb_projection_is_sum_of_components():
    # Model outputs match L20 avg so blending is a no-op — we can assert exact sum
    df = _player_df(10, pts=22, ast=5, reb=9.5)
    odds = {"Star": {"PTS+REB": {
        "line": 25.5, "over_price": -110, "under_price": -110, "bookmaker": "FD",
    }}}
    models = {"PTS": _FakeModel(22.0), "REB": _FakeModel(9.5)}
    props = generate_wnba_props(df, _getter(models), odds, synthesize_missing=False)
    assert len(props) == 1
    p = props[0]
    assert p.stat == "PTS+REB"
    assert p.projected == pytest.approx(31.5)  # 22 + 9.5
    assert p.pick == "OVER"    # 31.5 > 25.5


def test_pts_ast_reb_projection_uses_all_three_models():
    df = _player_df(10, pts=16, ast=6.5, reb=7.5)
    odds = {"Star": {"PTS+REB+AST": {
        "line": 30.5, "over_price": +120, "under_price": -140, "bookmaker": "DK",
    }}}
    models = {"PTS": _FakeModel(16.0), "AST": _FakeModel(6.5), "REB": _FakeModel(7.5)}
    props = generate_wnba_props(df, _getter(models), odds, synthesize_missing=False)
    assert len(props) == 1
    p = props[0]
    assert p.projected == pytest.approx(30.0)      # 16 + 6.5 + 7.5
    assert p.pick == "UNDER"   # 30 < 30.5


def test_missing_component_model_skips_combo_prop():
    df = _player_df(10, pts=20, ast=5, reb=8)
    odds = {"Star": {"PTS+AST": {
        "line": 25.5, "over_price": -110, "under_price": -110, "bookmaker": "FD",
    }}}
    # Only PTS model — AST model missing, so we can't project the combo
    models = {"PTS": _FakeModel(22.0)}
    props = generate_wnba_props(df, _getter(models), odds, synthesize_missing=False)
    assert props == []


def test_hit_prob_uses_summed_actuals_for_combos():
    # Games: PTS=20, AST=5, so PTS+AST=25 every game. Line 22.5, pick OVER.
    df = _player_df(10, pts=20, ast=5, reb=8)
    odds = {"Star": {"PTS+AST": {
        "line": 22.5, "over_price": -110, "under_price": -110, "bookmaker": "FD",
    }}}
    models = {"PTS": _FakeModel(21.0), "AST": _FakeModel(5.5)}
    props = generate_wnba_props(df, _getter(models), odds, synthesize_missing=False)
    p = props[0]
    assert p.hit_prob == 1.0  # 25 > 22.5 in every game
