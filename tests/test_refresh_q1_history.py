"""stats.nba.com throttles heavy callers: the 1Q history must fetch as little as it can."""
import os
import time
import pandas as pd
from scripts import refresh_q1_history as q1


def _rows(season):
    return pd.DataFrame([{column: 0 for column in q1.COLUMNS} | {'SEASON': season, 'GAME_ID': '0022600001'}])


def _setup(monkeypatch, tmp_path, saved_seasons):
    out = tmp_path / 'q1.csv'
    if saved_seasons:
        pd.concat([_rows(s) for s in saved_seasons]).to_csv(out, index=False)
    fetched = []
    monkeypatch.setattr(q1, 'OUT', out)
    monkeypatch.setattr(q1, 'current_season', lambda today: '2026-27')
    monkeypatch.setattr(q1, 'fetch', lambda season: fetched.append(season) or _rows(season))
    return out, fetched


def test_last_season_is_fetched_once_and_kept(monkeypatch, tmp_path):
    out, fetched = _setup(monkeypatch, tmp_path, ['2025-26'])
    os.utime(out, (time.time() - 2 * 86400,) * 2)
    q1.main()
    assert fetched == ['2026-27'], 'a finished season never changes'
    assert set(pd.read_csv(out)['SEASON']) == {'2025-26', '2026-27'}


def test_a_fresh_file_skips_the_network_entirely(monkeypatch, tmp_path):
    out, fetched = _setup(monkeypatch, tmp_path, ['2025-26', '2026-27'])
    q1.main()
    assert fetched == []


def test_a_first_run_fetches_both_seasons(monkeypatch, tmp_path):
    _, fetched = _setup(monkeypatch, tmp_path, [])
    q1.main()
    assert fetched == ['2025-26', '2026-27']
