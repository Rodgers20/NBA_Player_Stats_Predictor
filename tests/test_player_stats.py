# tests/test_player_stats.py

import pandas as pd
from unittest.mock import patch
from utils.data_fetch import get_player_stats


@patch("utils.data_fetch.playergamelog.PlayerGameLog")
def test_get_player_stats_structure(mock_log):
    mock_df = pd.DataFrame(
        {
            "GAME_DATE": ["2025-11-01"],
            "MATCHUP": ["LAL vs GSW"],
            "PTS": [28],
            "AST": [7],
            "REB": [9],
        }
    )

    mock_instance = mock_log.return_value
    mock_instance.get_data_frames.return_value = [mock_df]

    df = get_player_stats("LeBron James")

    assert isinstance(df, pd.DataFrame)
    assert "PTS" in df.columns
    assert "AST" in df.columns
    assert "REB" in df.columns


@patch("utils.data_fetch.playergamelog.PlayerGameLog")
def test_get_player_stats_no_data(mock_log):
    mock_instance = mock_log.return_value
    mock_instance.get_data_frames.return_value = [pd.DataFrame()]

    df = get_player_stats("LeBron James")
    assert df.empty
