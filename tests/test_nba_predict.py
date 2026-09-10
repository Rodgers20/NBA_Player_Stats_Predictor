"""Tests for utils.nba_predict — NBA next-game projections.

Mirrors the WNBA path (utils/wnba_predict.py + the blend/clip helpers in
utils/wnba_props.py) so the NBA player page can show the same Next Game
Prediction card. The blending guard matters: an unguarded model output can
project 1.0 REB for a 4.5-REB player, and this card is user-facing.
"""

import pandas as pd
import pytest

from utils.nba_predict import (
    NBA_STAT_CAPS,
    blend_projection,
    clip_prediction,
    get_tonight_matchup_for_player,
    project_next_game,
)


# ── clipping ──────────────────────────────────────────────────────────────────

def test_clip_rejects_negative():
    assert clip_prediction("PTS", -5.0) == 0.0


def test_clip_caps_at_nba_maximum():
    assert clip_prediction("PTS", 500.0) == NBA_STAT_CAPS["PTS"]
    assert clip_prediction("REB", 500.0) == NBA_STAT_CAPS["REB"]
    assert clip_prediction("AST", 500.0) == NBA_STAT_CAPS["AST"]


def test_clip_passes_through_normal_values():
    assert clip_prediction("PTS", 27.4) == pytest.approx(27.4)


def test_clip_handles_none():
    assert clip_prediction("PTS", None) == 0.0


def test_nba_caps_exceed_wnba_caps():
    """Guard against copying WNBA caps verbatim — NBA scoring runs higher."""
    from utils.wnba_props import _clip_prediction as wnba_clip
    assert clip_prediction("PTS", 60.0) == 60.0, "60 pts is a real NBA game"
    assert wnba_clip("PTS", 60.0) < 60.0, "sanity: WNBA cap is lower"


# ── blending ──────────────────────────────────────────────────────────────────

def test_blend_is_weighted_average():
    assert blend_projection(30.0, 20.0, "PTS", alpha=0.5) == pytest.approx(25.0)


def test_blend_floors_at_40_percent_of_form():
    """The catastrophic-under-projection guard.

    Needs a high alpha for the floor to actually bind: at the default 0.55 the
    form term alone (0.45 * 10 = 4.5) already clears the 4.0 floor.
    """
    assert blend_projection(0.1, 10.0, "REB", alpha=0.95) == pytest.approx(4.0)


def test_blend_caps_at_160_percent_of_form():
    assert blend_projection(999.0, 10.0, "PTS", alpha=0.55) == pytest.approx(16.0)


def test_blend_without_form_falls_back_to_clipped_model():
    assert blend_projection(25.0, 0.0, "PTS") == pytest.approx(25.0)
    assert blend_projection(-3.0, 0.0, "PTS") == 0.0


def test_blend_default_alpha_favours_model():
    """Model carries more weight than form when the form band is not binding."""
    out = blend_projection(30.0, 25.0, "PTS")          # band = [10, 40]
    assert out == pytest.approx(0.55 * 30 + 0.45 * 25)
    assert out > (30.0 + 25.0) / 2, "must lean toward the model, not a 50/50 mean"


def test_blend_ceiling_binds_before_alpha():
    """A model far above recent form is capped at 1.6x form, not averaged in."""
    assert blend_projection(30.0, 10.0, "PTS") == pytest.approx(16.0)


# ── matchup resolution ────────────────────────────────────────────────────────

_GAMES = [
    {"home": {"abbrev": "LAL"}, "away": {"abbrev": "BOS"}},
    {"home": {"abbrev": "GSW"}, "away": {"abbrev": "DEN"}},
]


def test_matchup_home_team():
    assert get_tonight_matchup_for_player("LAL", _GAMES) == ("BOS", True)


def test_matchup_away_team():
    assert get_tonight_matchup_for_player("BOS", _GAMES) == ("LAL", False)


def test_matchup_team_not_playing_returns_none():
    assert get_tonight_matchup_for_player("MIA", _GAMES) is None


def test_matchup_handles_empty_and_none():
    assert get_tonight_matchup_for_player("LAL", []) is None
    assert get_tonight_matchup_for_player("", _GAMES) is None


def test_matchup_is_case_insensitive():
    assert get_tonight_matchup_for_player("lal", _GAMES) == ("BOS", True)


def test_matchup_accepts_flat_game_shape():
    """utils.data_fetch.get_todays_games returns flat home/away strings."""
    flat = [{"home": "LAL", "away": "BOS"}]
    assert get_tonight_matchup_for_player("LAL", flat) == ("BOS", True)


# ── end-to-end projection ─────────────────────────────────────────────────────

def _history(n=25, pts=25.0, ast=6.0, reb=8.0):
    return pd.DataFrame([{
        "PLAYER_NAME": "Test Player",
        "TEAM_ABBREVIATION": "LAL",
        "GAME_DATE": f"Jan {(i % 28) + 1:02d}, 2026",
        "MATCHUP": "LAL vs. BOS",
        "SEASON": "2025-26",
        "PTS": pts, "AST": ast, "REB": reb, "MIN": 34.0,
        "FGA": 18.0, "FG_PCT": 0.47, "FG3A": 6.0, "FG3_PCT": 0.36,
    } for i in range(n)])


class _StubModel:
    def __init__(self, value):
        self.value = value
        self.feature_columns = ["rolling_avg_pts_5"]

    def predict(self, features):
        return self.value


def test_project_next_game_returns_all_three_stats():
    out = project_next_game(
        _history(), predictor_getter=lambda s: _StubModel(26.0),
        opponent="BOS", is_home=True,
    )
    assert set(out["stats"]) == {"PTS", "AST", "REB"}
    assert out["opponent"] == "BOS"
    assert out["is_home"] is True


def test_project_next_game_blends_toward_form():
    """A wild model output must be pulled back toward the player's L20."""
    out = project_next_game(
        _history(pts=25.0), predictor_getter=lambda s: _StubModel(200.0),
        opponent="BOS", is_home=True,
    )
    assert out["stats"]["PTS"] <= 25.0 * 1.6


def test_project_next_game_empty_history_returns_none():
    assert project_next_game(pd.DataFrame(), lambda s: _StubModel(1.0),
                             "BOS", True) is None


def test_project_next_game_survives_missing_model():
    out = project_next_game(_history(), predictor_getter=lambda s: None,
                            opponent="BOS", is_home=True)
    assert out is not None
    assert out["stats"] == {}


def test_project_next_game_survives_model_error():
    class _Boom:
        feature_columns = []

        def predict(self, f):
            raise RuntimeError("bad feature vector")

    out = project_next_game(_history(), predictor_getter=lambda s: _Boom(),
                            opponent="BOS", is_home=True)
    assert out is not None and out["stats"] == {}


def test_project_next_game_includes_reference_averages():
    """Subtitle needs L20 and L5 to let the user sanity-check the number."""
    out = project_next_game(_history(pts=20.0), lambda s: _StubModel(20.0),
                            "BOS", True)
    assert out["l20"]["PTS"] == pytest.approx(20.0)
    assert out["l5"]["PTS"] == pytest.approx(20.0)
