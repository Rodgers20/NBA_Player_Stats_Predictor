# tests/test_player_stats.py
"""Tests for player game log loading from Kaggle data."""

from pathlib import Path
import pandas as pd
from unittest.mock import patch
from utils.kaggle_loader import load_player_game_logs


@patch("utils.kaggle_loader._get_dataset_path")
def test_load_player_game_logs_structure(mock_path, tmp_path):
    """Verify loaded game logs have expected columns and types."""
    mock_path.return_value = tmp_path

    # Create minimal PlayerStatistics.csv
    player_stats = pd.DataFrame({
        "personId": [201566],
        "firstName": ["LeBron"],
        "lastName": ["James"],
        "gameId": ["0022500001"],
        "gameDateTimeEst": ["2025-10-28T00:00:00"],
        "playerteamCity": ["Los Angeles"],
        "playerteamName": ["Lakers"],
        "opponentteamCity": ["Golden State"],
        "opponentteamName": ["Warriors"],
        "home": [True],
        "win": [True],
        "numMinutes": [35.5],
        "points": [28],
        "assists": [7],
        "reboundsTotal": [9],
        "reboundsOffensive": [1],
        "reboundsDefensive": [8],
        "fieldGoalsMade": [11],
        "fieldGoalsAttempted": [20],
        "fieldGoalsPercentage": [0.55],
        "threePointersMade": [3],
        "threePointersAttempted": [8],
        "threePointersPercentage": [0.375],
        "freeThrowsMade": [3],
        "freeThrowsAttempted": [4],
        "freeThrowsPercentage": [0.75],
        "steals": [2],
        "blocks": [1],
        "turnovers": [3],
        "foulsPersonal": [2],
        "plusMinusPoints": [12],
        "gameType": ["Regular Season"],
    })
    player_stats.to_csv(tmp_path / "PlayerStatistics.csv", index=False)

    # Create minimal TeamHistories.csv
    team_hist = pd.DataFrame({
        "teamId": [1610612747, 1610612744],
        "teamCity": ["Los Angeles", "Golden State"],
        "teamName": ["Lakers", "Warriors"],
        "teamAbbrev": ["LAL", "GSW"],
        "seasonActiveTill": [0, 0],
        "league": ["NBA", "NBA"],
    })
    team_hist.to_csv(tmp_path / "TeamHistories.csv", index=False)

    df = load_player_game_logs(num_seasons=1)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "PTS" in df.columns
    assert "AST" in df.columns
    assert "REB" in df.columns
    assert "PLAYER_NAME" in df.columns
    assert "GAME_DATE" in df.columns
    assert "MATCHUP" in df.columns


@patch("utils.kaggle_loader._get_dataset_path")
def test_load_player_game_logs_empty(mock_path, tmp_path):
    """Verify empty DataFrame returned when no data matches."""
    mock_path.return_value = tmp_path

    # Create empty PlayerStatistics.csv with just headers
    pd.DataFrame(columns=[
        "personId", "firstName", "lastName", "gameId",
        "gameDateTimeEst", "playerteamCity", "playerteamName",
        "opponentteamCity", "opponentteamName", "home", "win",
        "numMinutes", "points", "assists", "reboundsTotal",
        "reboundsOffensive", "reboundsDefensive",
        "fieldGoalsMade", "fieldGoalsAttempted", "fieldGoalsPercentage",
        "threePointersMade", "threePointersAttempted", "threePointersPercentage",
        "freeThrowsMade", "freeThrowsAttempted", "freeThrowsPercentage",
        "steals", "blocks", "turnovers", "foulsPersonal",
        "plusMinusPoints", "gameType",
    ]).to_csv(tmp_path / "PlayerStatistics.csv", index=False)

    pd.DataFrame(columns=[
        "teamId", "teamCity", "teamName", "teamAbbrev",
        "seasonActiveTill", "league",
    ]).to_csv(tmp_path / "TeamHistories.csv", index=False)

    df = load_player_game_logs(num_seasons=1)
    assert df.empty
