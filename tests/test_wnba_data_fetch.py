"""Structural tests for utils.wnba_data_fetch (no network)."""

import types
import pandas as pd
import pytest

from utils import wnba_data_fetch


def _fake_scoreboardv3(monkeypatch, games_df, teams_df):
    """Monkeypatch scoreboardv3.ScoreboardV3 to return canned frames."""
    class _Fake:
        def __init__(self, *args, **kwargs):
            pass

        def get_data_frames(self):
            # Frame 0 = metadata (unused); Frame 1 = games; Frame 2 = team lines
            return [pd.DataFrame(), games_df, teams_df]

    # nba_api submodule is imported inside the function under test
    from nba_api.stats.endpoints import scoreboardv3 as sb_module
    monkeypatch.setattr(sb_module, "ScoreboardV3", _Fake)


def test_get_todays_wnba_games_parses_two_games(monkeypatch):
    games_df = pd.DataFrame([
        {"gameId": "1022600001", "gameStatus": 1, "gameStatusText": "7:00 pm ET",
         "gameEt": "2026-08-03T19:00:00Z", "homeTeamId": 1611661330},
        {"gameId": "1022600002", "gameStatus": 3, "gameStatusText": "Final",
         "gameEt": "2026-08-03T19:00:00Z", "homeTeamId": 1611661313},
    ])
    teams_df = pd.DataFrame([
        {"gameId": "1022600001", "teamId": 1611661330, "teamCity": "Atlanta", "teamName": "Dream",
         "teamTricode": "ATL", "wins": 18, "losses": 10, "score": 0},
        {"gameId": "1022600001", "teamId": 1611661319, "teamCity": "Las Vegas", "teamName": "Aces",
         "teamTricode": "LVA", "wins": 20, "losses": 9, "score": 0},
        {"gameId": "1022600002", "teamId": 1611661313, "teamCity": "New York", "teamName": "Liberty",
         "teamTricode": "NYL", "wins": 17, "losses": 13, "score": 88},
        {"gameId": "1022600002", "teamId": 1611661328, "teamCity": "Seattle", "teamName": "Storm",
         "teamTricode": "SEA", "wins": 6, "losses": 25, "score": 74},
    ])
    _fake_scoreboardv3(monkeypatch, games_df, teams_df)

    games = wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert len(games) == 2

    g1 = games[0]
    assert g1["home"]["abbrev"] == "ATL"
    assert g1["away"]["abbrev"] == "LVA"
    assert g1["home"]["wins"] == 18
    assert g1["away"]["losses"] == 9
    assert g1["status"] == "Scheduled"

    g2 = games[1]
    assert g2["home"]["abbrev"] == "NYL"
    assert g2["away"]["abbrev"] == "SEA"
    assert g2["home"]["score"] == 88
    assert g2["away"]["score"] == 74
    assert g2["status"] == "Final"


def test_get_todays_wnba_games_empty_on_no_games(monkeypatch):
    _fake_scoreboardv3(monkeypatch, pd.DataFrame(), pd.DataFrame())
    assert wnba_data_fetch.get_todays_wnba_games("2026-01-01") == []


def test_get_teams_playing_today_wnba(monkeypatch):
    games_df = pd.DataFrame([
        {"gameId": "1022600001", "gameStatus": 1, "gameStatusText": "7:00 pm ET",
         "gameEt": "2026-08-03T19:00:00Z", "homeTeamId": 1611661330},
    ])
    teams_df = pd.DataFrame([
        {"gameId": "1022600001", "teamId": 1611661330, "teamCity": "Atlanta", "teamName": "Dream",
         "teamTricode": "ATL", "wins": 18, "losses": 10, "score": 0},
        {"gameId": "1022600001", "teamId": 1611661319, "teamCity": "Las Vegas", "teamName": "Aces",
         "teamTricode": "LVA", "wins": 20, "losses": 9, "score": 0},
    ])
    _fake_scoreboardv3(monkeypatch, games_df, teams_df)
    playing = wnba_data_fetch.get_teams_playing_today_wnba("2026-08-03")
    assert playing == {"ATL", "LVA"}
