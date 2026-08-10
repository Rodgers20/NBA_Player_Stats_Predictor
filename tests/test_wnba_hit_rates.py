"""Tests for utils.wnba_hit_rates."""

import pandas as pd
import pytest

from utils.wnba_hit_rates import compute_hit_rates, _best_thresholds_for_stat


def _make_player(name, team, pts_list, reb_list=None, ast_list=None, fg3m_list=None, min_avg=25):
    n = len(pts_list)
    reb_list = reb_list or [0] * n
    ast_list = ast_list or [0] * n
    fg3m_list = fg3m_list or [0] * n
    return pd.DataFrame({
        "PLAYER_NAME": [name] * n,
        "TEAM_ABBREVIATION": [team] * n,
        "_date": pd.date_range("2026-07-01", periods=n),
        "MIN": [min_avg] * n,
        "PTS": pts_list, "REB": reb_list, "AST": ast_list, "FG3M": fg3m_list,
    })


def test_best_thresholds_picks_highest_tier_per_hit_count():
    # Player scores [20,21,22,23,24,25,26,27,28,29] — 10 games spread 20-29
    values = pd.Series([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    tiers = _best_thresholds_for_stat(values, "PTS", min_hits=7)
    # 20+ Pts 10/10 must be present (all 10 games clear 20)
    ratios = {(t, h) for (t, h, _n) in tiers}
    assert (20, 10) in ratios
    # With min_hits=7, max threshold with hits>=7 is 23 (values 23..29 = 7 games)
    max_threshold = max(t for t, _h, _n in tiers)
    assert max_threshold >= 23


def test_best_thresholds_skips_trivial_floors():
    # Player scores [2,3,4,5,6,7,8,9,10,11] — dominated by low values
    values = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    tiers = _best_thresholds_for_stat(values, "PTS", min_hits=7)
    # Floor is 8 PTS for the Points stat — should never return a threshold below 8
    for t, _h, _n in tiers:
        assert t >= 8


def test_compute_hit_rates_only_includes_tonights_players():
    df = pd.concat([
        _make_player("Starter A", "LVA", [20] * 10),
        _make_player("Starter B", "IND", [22] * 10),
        _make_player("Off-Day Player", "PHX", [30] * 10),
    ])
    games = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "IND"}}]
    result = compute_hit_rates(df, games)
    assert len(result) == 1
    names = {e.player_name for e in result[0]["entries"]}
    assert "Starter A" in names
    assert "Starter B" in names
    assert "Off-Day Player" not in names


def test_compute_hit_rates_filters_bench_players():
    # Bench player with high scoring games but avg 5 min — should be excluded
    df = pd.concat([
        _make_player("Starter", "LVA", [15] * 10, min_avg=28),
        _make_player("Bench Only", "LVA", [15] * 10, min_avg=5),
    ])
    games = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "IND"}}]
    result = compute_hit_rates(df, games, min_avg_min=15.0)
    names = {e.player_name for e in result[0]["entries"]}
    assert "Starter" in names
    assert "Bench Only" not in names


def test_compute_hit_rates_requires_n_games():
    # Only 6 games — below n_games=10 default
    df = _make_player("Short History", "LVA", [20] * 6)
    games = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "IND"}}]
    result = compute_hit_rates(df, games, n_games=10)
    assert result == [] or all(len(gr["entries"]) == 0 for gr in result)


def test_result_entries_sorted_by_ratio_desc():
    df = pd.concat([
        _make_player("Perfect", "LVA", [30] * 10),   # 20+ Pts 10/10
        _make_player("Solid",   "LVA", [30] * 8 + [10, 10]),  # 20+ Pts 8/10
    ])
    games = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "IND"}}]
    result = compute_hit_rates(df, games)
    entries = result[0]["entries"]
    # First entry should have the higher ratio
    assert entries[0].ratio >= entries[-1].ratio
