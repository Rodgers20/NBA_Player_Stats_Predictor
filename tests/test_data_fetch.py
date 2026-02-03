# tests/test_data_fetch.py

import pandas as pd
from unittest.mock import patch
from utils.data_fetch import get_team_data


@patch("utils.data_fetch.leaguedashteamstats.LeagueDashTeamStats")
def test_get_team_data_structure(mock_stats):
    # Mock DataFrame returned by nba_api
    mock_df = pd.DataFrame(
        {
            "TEAM_ID": [1610612737],
            "TEAM_NAME": ["Atlanta Hawks"],
            "GP": [10],
            "PTS": [115.2],
            "REB": [44.3],
        }
    )

    mock_instance = mock_stats.return_value
    mock_instance.get_data_frames.return_value = [mock_df]

    df = get_team_data()

    # The output should be a DataFrame and contain key columns from nba_api
    assert isinstance(df, pd.DataFrame)
    assert "TEAM_ID" in df.columns
    assert "TEAM_NAME" in df.columns
    assert "PTS" in df.columns


@patch("utils.data_fetch.leaguedashteamstats.LeagueDashTeamStats")
def test_get_team_data_empty(mock_stats):
    # Simulate api returning no data
    mock_instance = mock_stats.return_value
    mock_instance.get_data_frames.return_value = [pd.DataFrame()]

    df = get_team_data()
    assert df.empty
