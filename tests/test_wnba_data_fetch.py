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


def test_blocked_stats_service_falls_back_to_espn_and_is_skipped_afterwards(monkeypatch):
    calls = []

    class _Blocked:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs.get("timeout"))
            raise TimeoutError("stats.nba.com blocked")

    from nba_api.stats.endpoints import scoreboardv3 as sb_module
    from utils import basketball_content
    monkeypatch.setattr(sb_module, "ScoreboardV3", _Blocked)
    competitor = lambda side, abbr, name, score: dict(
        homeAway=side, score=score, team=dict(id="1", abbreviation=abbr, displayName=name),
        records=[dict(type="total", summary="10-4")])
    board = dict(events=[dict(id="401", date="2026-08-03T23:00Z", competitions=[dict(
        competitors=[competitor("home", "NYL", "New York Liberty", "0"),
                     competitor("away", "LVA", "Las Vegas Aces", "0")],
        status=dict(type=dict(state="pre", shortDetail="8/3 - 7:00 PM EDT")))])])
    monkeypatch.setattr(basketball_content, "_fetch", lambda league, path, **params: board)
    monkeypatch.setattr(wnba_data_fetch, "_stats_down_until", 0.0)
    wnba_data_fetch.clear_schedule_cache()

    games = wnba_data_fetch.get_todays_wnba_games("2026-08-03")
    assert [(g["home"]["abbrev"], g["away"]["abbrev"], g["home"]["wins"], g["home"]["losses"])
            for g in games] == [("NYL", "LVA", 10, 4)]
    assert games[0]["game_id"] == "401" and games[0]["status_text"]
    assert calls and calls[0] <= 10                      # bounded wait on the blocked service
    wnba_data_fetch.get_todays_wnba_games("2026-08-04")
    assert len(calls) == 1                               # stats service skipped while it is down


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
