"""Tests for utils.wnba_loader.

These tests verify structure/logic without hitting the network; a live
integration test hitting nba_api is deliberately skipped in the default suite.
"""

import pandas as pd
import pytest

from utils.wnba_loader import (
    _normalize_player_logs,
    load_player_positions,
    get_recent_wnba_seasons,
)


def test_get_recent_wnba_seasons_returns_str_years():
    seasons = get_recent_wnba_seasons(3)
    assert len(seasons) == 3
    for s in seasons:
        assert isinstance(s, str)
        assert len(s) == 4
        assert s.isdigit()
    # Descending
    assert seasons == sorted(seasons, reverse=True)


def test_normalize_player_logs_renames_and_reformats():
    raw = pd.DataFrame({
        "SEASON_ID": ["22025", "22025"],
        "PLAYER_ID": [1001, 1002],
        "PLAYER_NAME": ["Player A", "Player B"],
        "TEAM_ID": [1611661322, 1611661313],
        "TEAM_ABBREVIATION": ["LVA", "NYL"],
        "TEAM_NAME": ["Las Vegas Aces", "New York Liberty"],
        "GAME_ID": ["0022500001", "0022500002"],
        "GAME_DATE": ["2025-05-16", "2025-05-17"],
        "MATCHUP": ["LVA vs. NYL", "NYL @ LVA"],
        "WL": ["W", "L"],
        "MIN": [30.0, 25.0],
        "FGM": [10, 8], "FGA": [20, 15], "FG_PCT": [0.5, 0.53],
        "FG3M": [2, 1], "FG3A": [5, 3], "FG3_PCT": [0.4, 0.33],
        "FTM": [5, 4], "FTA": [5, 4], "FT_PCT": [1.0, 1.0],
        "OREB": [2, 1], "DREB": [7, 5], "REB": [9, 6],
        "AST": [5, 3], "STL": [2, 1], "BLK": [1, 0], "TOV": [3, 2], "PF": [2, 3],
        "PTS": [27, 21], "PLUS_MINUS": [8, -8],
        "FANTASY_PTS": [45.5, 33.0],
        "VIDEO_AVAILABLE": [1, 1],
    })

    df = _normalize_player_logs(raw)

    # Column renames
    assert "Player_ID" in df.columns
    assert "Game_ID" in df.columns
    assert "PLAYER_ID" not in df.columns
    assert "GAME_ID" not in df.columns
    # FANTASY_PTS dropped
    assert "FANTASY_PTS" not in df.columns
    # Date reformatted to "Mmm dd, YYYY"
    assert df["GAME_DATE"].iloc[0] == "May 16, 2025"
    assert df["GAME_DATE"].iloc[1] == "May 17, 2025"


def test_load_player_positions_infers_positions():
    # Build synthetic logs: a guard (high AST, low REB), a center (high REB),
    # and a forward (mid-range on both).
    logs = pd.DataFrame({
        "Player_ID": [1, 1, 2, 2, 3, 3],
        "PLAYER_NAME": ["Guard G", "Guard G", "Center C", "Center C", "Forward F", "Forward F"],
        "TEAM_ABBREVIATION": ["LVA"] * 6,
        "SEASON": ["2025"] * 6,
        "GAME_DATE": ["May 16, 2025", "May 17, 2025"] * 3,
        "AST": [8, 9, 1, 2, 3, 4],
        "REB": [3, 2, 10, 12, 5, 6],
    })

    positions = load_player_positions(logs)

    assert set(positions.columns) == {"PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "POSITION", "SEASON"}
    pos_by_pid = dict(zip(positions["PLAYER_ID"], positions["POSITION"]))
    assert pos_by_pid[1] == "G"   # 8.5 avg AST >= 3.5 threshold, 2.5 REB < 5
    assert pos_by_pid[2] == "C"   # 11 avg REB >= 6.5 threshold
    assert pos_by_pid[3] == "F"   # falls through to default
