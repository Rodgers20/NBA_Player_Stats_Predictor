"""Tests for utils.wnba_predict (tonight-context feature builder)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from utils.wnba_predict import (
    build_tonight_feature_row,
    get_tonight_matchup_for_player,
)


def _player_history(recent_days_ago: int = 1):
    """Player history with one row `recent_days_ago` days before today."""
    last_date = pd.Timestamp(date.today() - timedelta(days=recent_days_ago))
    return pd.DataFrame({
        "PLAYER_NAME": ["Star"],
        "TEAM_ABBREVIATION": ["LVA"],
        "SEASON": ["2026"],
        "_date": [last_date],
        "POSITION": ["F"],
        "PTS": [20], "AST": [3], "REB": [8],
        "is_home": [0],
        "days_rest": [4],
        "is_back_to_back": [0],
        "opp_def_pts_allowed": [999],  # will be overridden
        "opp_pts_def_rank": [7],
        "opp_ast_def_rank": [7],
        "opp_reb_def_rank": [7],
        "opp_elite_defense": [0],
        "opp_good_defense": [0],
        "opp_poor_defense": [0],
        "team_pace": [95.0],
    })


def test_home_and_rest_days_overrides():
    ph = _player_history(recent_days_ago=2)
    feat = build_tonight_feature_row(
        ph, tonight_opponent="ATL", is_home=True,
    )
    assert feat["is_home"] == 1
    assert feat["days_rest"] == 2
    assert feat["is_back_to_back"] == 0


def test_back_to_back_flag_set_on_one_day_rest():
    ph = _player_history(recent_days_ago=1)
    feat = build_tonight_feature_row(
        ph, tonight_opponent="ATL", is_home=False,
    )
    assert feat["is_back_to_back"] == 1
    assert feat["days_rest"] == 1
    assert feat["is_home"] == 0


def test_opponent_defense_stats_override():
    ph = _player_history()
    team_def = pd.DataFrame({
        "TEAM_ABBREVIATION": ["ATL", "NYL"],
        "SEASON": ["2026", "2026"],
        "OPP_PTS": [78.5, 88.0],
        "OPP_FG_PCT": [0.41, 0.47],
        "OPP_FG3_PCT": [0.32, 0.38],
    })
    feat = build_tonight_feature_row(
        ph, tonight_opponent="ATL", is_home=True, team_def=team_def,
    )
    assert feat["opp_def_pts_allowed"] == pytest.approx(78.5)
    assert feat["opp_def_fg_pct"] == pytest.approx(0.41)


def test_defense_vs_position_updates_ranks_and_tiers():
    ph = _player_history()
    dvp = pd.DataFrame({
        "TEAM_ABBREVIATION": ["ATL", "NYL"],
        "SEASON": ["2026", "2026"],
        "POSITION": ["F", "F"],
        "PTS_RANK": [2, 12],   # ATL = elite defense vs forwards; NYL = poor
        "AST_RANK": [5, 8],
        "REB_RANK": [3, 10],
    })
    # Tough matchup
    feat = build_tonight_feature_row(ph, "ATL", True, def_vs_pos=dvp)
    assert feat["opp_pts_def_rank"] == 2
    assert feat["opp_elite_defense"] == 1
    assert feat["opp_poor_defense"] == 0

    # Soft matchup
    feat2 = build_tonight_feature_row(ph, "NYL", False, def_vs_pos=dvp)
    assert feat2["opp_pts_def_rank"] == 12
    assert feat2["opp_poor_defense"] == 1
    assert feat2["opp_elite_defense"] == 0


def test_get_tonight_matchup_finds_home_and_away():
    games = [
        {"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}},
        {"home": {"abbrev": "NYL"}, "away": {"abbrev": "SEA"}},
    ]
    assert get_tonight_matchup_for_player("LVA", games) == ("ATL", True)
    assert get_tonight_matchup_for_player("ATL", games) == ("LVA", False)
    assert get_tonight_matchup_for_player("SEA", games) == ("NYL", False)


def test_get_tonight_matchup_returns_none_when_off():
    games = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    assert get_tonight_matchup_for_player("CHI", games) is None
