# tests/test_data_fetch.py
"""Tests for team data loading from Kaggle data."""

from pathlib import Path
import pandas as pd
from unittest.mock import patch
from utils.kaggle_loader import load_team_stats


@patch("utils.kaggle_loader._get_dataset_path")
def test_load_team_stats_structure(mock_path, tmp_path):
    """Verify loaded team stats have expected columns."""
    mock_path.return_value = tmp_path

    # Create minimal TeamStatistics.csv
    team_stats = pd.DataFrame({
        "gameId": ["0022500001"],
        "teamId": [1610612737],
        "teamCity": ["Atlanta"],
        "teamName": ["Hawks"],
        "opponentTeamId": [1610612738],
        "home": [True],
        "win": [True],
        "teamScore": [115],
        "opponentScore": [108],
        "assists": [25],
        "blocks": [5],
        "steals": [8],
        "fieldGoalsMade": [42],
        "fieldGoalsAttempted": [88],
        "fieldGoalsPercentage": [0.477],
        "threePointersMade": [12],
        "threePointersAttempted": [35],
        "threePointersPercentage": [0.343],
        "freeThrowsMade": [19],
        "freeThrowsAttempted": [24],
        "freeThrowsPercentage": [0.792],
        "reboundsDefensive": [33],
        "reboundsOffensive": [10],
        "reboundsTotal": [43],
        "turnovers": [13],
        "foulsPersonal": [20],
        "plusMinusPoints": [7],
        "numMinutes": [240],
        "gameType": ["Regular Season"],
        "gameDateTimeEst": ["2025-10-28T00:00:00"],
    })
    team_stats.to_csv(tmp_path / "TeamStatistics.csv", index=False)

    # Create minimal Games.csv for season filtering
    games = pd.DataFrame({
        "gameId": ["0022500001"],
        "gameDateTimeEst": ["2025-10-28T00:00:00"],
        "gameType": ["Regular Season"],
    })
    games.to_csv(tmp_path / "Games.csv", index=False)

    # Create minimal TeamHistories.csv
    team_hist = pd.DataFrame({
        "teamId": [1610612737, 1610612738],
        "teamCity": ["Atlanta", "Boston"],
        "teamName": ["Hawks", "Celtics"],
        "teamAbbrev": ["ATL", "BOS"],
        "seasonActiveTill": [0, 0],
        "league": ["NBA", "NBA"],
    })
    team_hist.to_csv(tmp_path / "TeamHistories.csv", index=False)

    df = load_team_stats(num_seasons=1)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "TEAM_ID" in df.columns
    assert "TEAM_NAME" in df.columns
    assert "PTS" in df.columns


@patch("utils.kaggle_loader._get_dataset_path")
def test_load_team_stats_empty(mock_path, tmp_path):
    """Verify empty DataFrame when no data matches."""
    mock_path.return_value = tmp_path

    pd.DataFrame(columns=[
        "gameId", "teamId", "teamCity", "teamName", "opponentTeamId",
        "home", "win", "teamScore", "opponentScore", "assists", "blocks",
        "steals", "fieldGoalsMade", "fieldGoalsAttempted", "fieldGoalsPercentage",
        "threePointersMade", "threePointersAttempted", "threePointersPercentage",
        "freeThrowsMade", "freeThrowsAttempted", "freeThrowsPercentage",
        "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
        "turnovers", "foulsPersonal", "plusMinusPoints", "numMinutes",
        "gameType", "gameDateTimeEst",
    ]).to_csv(tmp_path / "TeamStatistics.csv", index=False)

    pd.DataFrame(columns=[
        "gameId", "gameDateTimeEst", "gameType",
    ]).to_csv(tmp_path / "Games.csv", index=False)

    pd.DataFrame(columns=[
        "teamId", "teamCity", "teamName", "teamAbbrev",
        "seasonActiveTill", "league",
    ]).to_csv(tmp_path / "TeamHistories.csv", index=False)

    df = load_team_stats(num_seasons=1)
    assert df.empty
