from datetime import date
import pandas as pd
import pytest
from scripts import refresh_nba_history


def test_season_rollover():
    assert refresh_nba_history.current_season(date(2026, 9, 30)) == '2025-26'
    assert refresh_nba_history.current_season(date(2026, 10, 1)) == '2026-27'
    assert refresh_nba_history.season_start('2026-27') == date(2026, 10, 1)


def test_refresh_uses_short_window_and_normalizes_provider_logs(monkeypatch):
    calls = []
    class FakeLog:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def get_data_frames(self):
            return [pd.DataFrame([{'PLAYER_ID': 1, 'GAME_ID': '10', 'PLAYER_NAME': 'A',
                'GAME_DATE': '2026-10-03', 'MATCHUP': 'A vs. B', 'PTS': 20}])]
    from nba_api.stats.endpoints import leaguegamelog
    monkeypatch.setattr(leaguegamelog, 'LeagueGameLog', FakeLog)
    rows = refresh_nba_history.fetch_recent('2026-27', date(2026, 10, 1))
    assert len(calls) == 4
    assert all(call['date_from_nullable'] == '10/01/2026' for call in calls)
    assert rows['Game_ID'].iloc[0] == '10'
    assert rows['SEASON'].iloc[0] == '2026-27'


def test_provider_failure_does_not_wipe_existing_history(monkeypatch):
    from nba_api.stats.endpoints import leaguegamelog
    def fail(**kwargs):
        raise TimeoutError('provider unavailable')
    monkeypatch.setattr(leaguegamelog, 'LeagueGameLog', fail)
    with pytest.raises(RuntimeError, match='existing history unchanged'):
        refresh_nba_history.fetch_recent('2026-27', date(2026, 10, 1))
