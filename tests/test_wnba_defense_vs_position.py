"""Tests for utils.wnba_loader.load_defense_vs_position."""

import pandas as pd
from utils.wnba_loader import load_defense_vs_position


def _make_logs(rows):
    return pd.DataFrame(rows)


def _make_positions(rows):
    return pd.DataFrame(rows)


def test_empty_inputs_return_empty_frame():
    assert load_defense_vs_position(pd.DataFrame(), pd.DataFrame()).empty


def test_aggregates_stats_by_opponent_and_position():
    logs = _make_logs([
        # Guard scoring against ATL
        {"Player_ID": 1, "PLAYER_NAME": "PG", "SEASON": "2026", "MATCHUP": "LVA @ ATL",
         "Game_ID": "G1", "PTS": 20, "AST": 6, "REB": 3, "FG3M": 2},
        {"Player_ID": 1, "PLAYER_NAME": "PG", "SEASON": "2026", "MATCHUP": "LVA @ ATL",
         "Game_ID": "G2", "PTS": 30, "AST": 8, "REB": 5, "FG3M": 3},
        # Center scoring against ATL
        {"Player_ID": 2, "PLAYER_NAME": "C", "SEASON": "2026", "MATCHUP": "LVA @ ATL",
         "Game_ID": "G3", "PTS": 25, "AST": 2, "REB": 15, "FG3M": 0},
    ])
    positions = _make_positions([
        {"PLAYER_ID": 1, "SEASON": "2026", "POSITION": "G"},
        {"PLAYER_ID": 2, "SEASON": "2026", "POSITION": "C"},
    ])
    df = load_defense_vs_position(logs, positions, seasons=["2026"])

    # ATL should have entries for both G and C
    atl_rows = df[df["TEAM_ABBREVIATION"] == "ATL"]
    positions_seen = set(atl_rows["POSITION"])
    assert {"G", "C"}.issubset(positions_seen)

    g_row = atl_rows[atl_rows["POSITION"] == "G"].iloc[0]
    assert g_row["PTS"] == 25.0  # (20 + 30) / 2
    assert g_row["AST"] == 7.0
    assert g_row["GP"] == 2

    c_row = atl_rows[atl_rows["POSITION"] == "C"].iloc[0]
    assert c_row["REB"] == 15.0


def test_produces_rank_columns_scoped_by_season_and_position():
    logs = _make_logs([
        # PG scoring against ATL (average 20 PTS)
        {"Player_ID": 1, "PLAYER_NAME": "PG", "SEASON": "2026", "MATCHUP": "LVA @ ATL",
         "Game_ID": "G1", "PTS": 20, "AST": 5, "REB": 3, "FG3M": 1},
        # PG scoring against NYL (average 30 PTS → weaker defense)
        {"Player_ID": 1, "PLAYER_NAME": "PG", "SEASON": "2026", "MATCHUP": "LVA @ NYL",
         "Game_ID": "G2", "PTS": 30, "AST": 5, "REB": 3, "FG3M": 1},
    ])
    positions = _make_positions([
        {"PLAYER_ID": 1, "SEASON": "2026", "POSITION": "G"},
    ])
    df = load_defense_vs_position(logs, positions, seasons=["2026"])

    atl = df[(df["TEAM_ABBREVIATION"] == "ATL") & (df["POSITION"] == "G")].iloc[0]
    nyl = df[(df["TEAM_ABBREVIATION"] == "NYL") & (df["POSITION"] == "G")].iloc[0]

    # ATL allows fewer PTS -> better defense -> rank 1
    assert atl["PTS_RANK"] == 1
    assert nyl["PTS_RANK"] == 2
