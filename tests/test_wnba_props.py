"""Tests for utils.wnba_props (no network, no real models)."""

import pandas as pd
import pytest

from utils.wnba_props import (
    american_to_decimal,
    american_to_implied_prob,
    generate_wnba_props,
    props_by_stat,
    _synthetic_line_from_recent,
    _blend_projection,
    SYNTHETIC_BOOKMAKER,
)


class _FakeModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, feat_row):
        return {"predicted_value": self.value}


def _fake_getter(preds):
    def get(stat):
        return preds.get(stat)
    return get


def test_american_conversions():
    # +100 → decimal 2.0, implied 50%
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_implied_prob(100) == pytest.approx(0.5)
    # -110 → decimal ~1.909, implied ~52.4%
    assert american_to_decimal(-110) == pytest.approx(1.9091, abs=0.001)
    assert american_to_implied_prob(-110) == pytest.approx(0.5238, abs=0.001)
    # None → sensible defaults
    assert american_to_decimal(None) == pytest.approx(1.91)
    assert american_to_implied_prob(None) == 0.5


def test_generate_props_over_pick_with_positive_edge():
    # Player who consistently scores 25+, line 20.5 -> should be OVER
    df = pd.DataFrame({
        "PLAYER_NAME": ["Star Player"] * 10,
        "TEAM_ABBREVIATION": ["LVA"] * 10,
        "_date": pd.date_range("2026-07-01", periods=10),
        "PTS": [30, 28, 25, 27, 22, 29, 26, 24, 31, 28],
        "AST": [3] * 10, "REB": [8] * 10, "FG3M": [1] * 10,
    })
    odds = {
        "Star Player": {"PTS": {"line": 20.5, "over_price": -110, "under_price": -110, "bookmaker": "FanDuel"}},
    }
    preds = {"PTS": _FakeModel(27.0)}

    props = generate_wnba_props(df, _fake_getter(preds), odds)
    assert len(props) == 1
    p = props[0]
    assert p.pick == "OVER"
    assert p.edge == pytest.approx(6.5)     # 27.0 - 20.5
    assert p.hit_prob == 1.0                # all 10 games above 20.5
    assert p.ev > 0
    assert p.confidence == "HIGH"


def test_generate_props_under_pick_with_negative_edge():
    # Model projects way under line
    df = pd.DataFrame({
        "PLAYER_NAME": ["Cold Player"] * 10,
        "TEAM_ABBREVIATION": ["NYL"] * 10,
        "_date": pd.date_range("2026-07-01", periods=10),
        "PTS": [8, 6, 10, 5, 9, 7, 4, 11, 6, 8],
        "AST": [2] * 10, "REB": [3] * 10, "FG3M": [0] * 10,
    })
    odds = {
        "Cold Player": {"PTS": {"line": 15.5, "over_price": +130, "under_price": -160, "bookmaker": "DK"}},
    }
    preds = {"PTS": _FakeModel(7.5)}

    props = generate_wnba_props(df, _fake_getter(preds), odds)
    assert len(props) == 1
    p = props[0]
    assert p.pick == "UNDER"
    assert p.hit_prob == 1.0


def test_skips_players_with_insufficient_history():
    df = pd.DataFrame({
        "PLAYER_NAME": ["New Player"] * 2,
        "TEAM_ABBREVIATION": ["GSV"] * 2,
        "_date": pd.date_range("2026-08-01", periods=2),
        "PTS": [15, 20], "AST": [3, 4], "REB": [5, 6], "FG3M": [1, 1],
    })
    odds = {"New Player": {"PTS": {"line": 15.5, "over_price": -110, "under_price": -110, "bookmaker": "FD"}}}
    props = generate_wnba_props(df, _fake_getter({"PTS": _FakeModel(18.0)}), odds, min_recent_games=5)
    assert props == []


def test_only_active_tonight_filter_drops_off_day_teams():
    """Props for players whose team isn't playing tonight should be excluded."""
    df = pd.DataFrame({
        "PLAYER_NAME": ["A"] * 10 + ["B"] * 10,
        "TEAM_ABBREVIATION": ["LVA"] * 10 + ["CHI"] * 10,   # A on LVA, B on CHI
        "_date": list(pd.date_range("2026-07-01", periods=10)) * 2,
        "PTS": [20] * 20, "AST": [3] * 20, "REB": [6] * 20, "FG3M": [1] * 20,
    })
    odds = {
        "A": {"PTS": {"line": 15.5, "over_price": -110, "under_price": -110, "bookmaker": "FD"}},
        "B": {"PTS": {"line": 15.5, "over_price": -110, "under_price": -110, "bookmaker": "FD"}},
    }
    preds = {"PTS": _FakeModel(18.0)}
    # Only LVA plays tonight — CHI player should be filtered out
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    props = generate_wnba_props(df, _fake_getter(preds), odds, todays_games=tonight)
    assert len(props) == 1
    assert props[0].player_name == "A"


def test_only_active_tonight_can_be_disabled():
    df = pd.DataFrame({
        "PLAYER_NAME": ["A"] * 10,
        "TEAM_ABBREVIATION": ["CHI"] * 10,
        "_date": pd.date_range("2026-07-01", periods=10),
        "PTS": [20] * 10, "AST": [3] * 10, "REB": [6] * 10, "FG3M": [1] * 10,
    })
    odds = {"A": {"PTS": {"line": 15.5, "over_price": -110, "under_price": -110, "bookmaker": "FD"}}}
    preds = {"PTS": _FakeModel(18.0)}
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    props = generate_wnba_props(df, _fake_getter(preds), odds, todays_games=tonight, only_active_tonight=False)
    assert len(props) == 1


def test_props_by_stat_grouping():
    df = pd.DataFrame({
        "PLAYER_NAME": ["Player A"] * 10,
        "TEAM_ABBREVIATION": ["ATL"] * 10,
        "_date": pd.date_range("2026-07-01", periods=10),
        "PTS": [20] * 10, "AST": [5] * 10, "REB": [8] * 10, "FG3M": [2] * 10,
    })
    odds = {
        "Player A": {
            "PTS": {"line": 15.5, "over_price": -110, "under_price": -110, "bookmaker": "FD"},
            "AST": {"line": 4.5,  "over_price": -110, "under_price": -110, "bookmaker": "FD"},
        }
    }
    preds = {"PTS": _FakeModel(22.0), "AST": _FakeModel(6.0)}

    props = generate_wnba_props(df, _fake_getter(preds), odds, synthesize_missing=False)
    grouped = props_by_stat(props)
    assert set(grouped) == {"PTS", "AST"}
    assert len(grouped["PTS"]) == 1
    assert len(grouped["AST"]) == 1


# --- Synthetic line tests --------------------------------------------------


def _starter_history(pts=18, ast=4, reb=7, n=12, avg_min=28):
    return pd.DataFrame({
        "PLAYER_NAME": ["Star"] * n,
        "TEAM_ABBREVIATION": ["LVA"] * n,
        "_date": pd.date_range("2026-07-01", periods=n),
        "MIN": [avg_min] * n,
        "PTS": [pts] * n, "AST": [ast] * n, "REB": [reb] * n, "FG3M": [1] * n,
    })


def test_synthetic_line_rounds_to_half_point_and_avoids_pushes():
    df = _starter_history(pts=18, n=12)
    line = _synthetic_line_from_recent(df, "PTS")
    # 18.0 avg → integer, must nudge to 17.5 or 18.5 to avoid a push
    assert line in {17.5, 18.5}


def test_synthetic_line_requires_min_games():
    df = _starter_history(pts=20, n=5)
    assert _synthetic_line_from_recent(df, "PTS") is None


def test_synthetic_line_skips_bench_players():
    # Player averaging 3 PTS — below the 6.5 meaningfulness floor
    df = _starter_history(pts=3, n=12)
    assert _synthetic_line_from_recent(df, "PTS") is None


def test_generate_props_synthesizes_when_odds_empty():
    df = _starter_history(pts=18, ast=6, reb=8, n=15)
    preds = {"PTS": _FakeModel(20.0), "AST": _FakeModel(7.0), "REB": _FakeModel(9.0),
             "FG3M": _FakeModel(1.5)}
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    props = generate_wnba_props(df, _fake_getter(preds), odds={},
                                todays_games=tonight, synthesize_missing=True)
    assert len(props) >= 3   # at least PTS, AST, REB
    assert all(p.bookmaker == SYNTHETIC_BOOKMAKER for p in props)
    stats_seen = {p.stat for p in props}
    assert {"PTS", "AST", "REB"}.issubset(stats_seen)


def test_generate_props_synthesize_can_be_disabled():
    df = _starter_history(pts=18, n=12)
    preds = {"PTS": _FakeModel(20.0)}
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    props = generate_wnba_props(df, _fake_getter(preds), odds={},
                                todays_games=tonight, synthesize_missing=False)
    assert props == []


def test_generate_props_prefers_real_odds_over_synthetic():
    df = _starter_history(pts=18, n=12)
    preds = {"PTS": _FakeModel(22.0), "AST": _FakeModel(6.0), "REB": _FakeModel(8.0), "FG3M": _FakeModel(1.0)}
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    # Real odds only cover PTS; AST/REB should fall back to synthetic
    odds = {"Star": {"PTS": {"line": 19.5, "over_price": -110, "under_price": -110, "bookmaker": "FanDuel"}}}
    props = generate_wnba_props(df, _fake_getter(preds), odds,
                                todays_games=tonight, synthesize_missing=True)
    by_stat = {p.stat: p for p in props}
    assert by_stat["PTS"].bookmaker == "FanDuel"
    assert by_stat["AST"].bookmaker == SYNTHETIC_BOOKMAKER
    assert by_stat["REB"].bookmaker == SYNTHETIC_BOOKMAKER


# --- Phase 9 overhaul tests -----------------------------------------------


def test_synthetic_line_uses_median_not_mean():
    """Player with occasional big games should get a line based on median, not the inflated mean."""
    # 12 games: 10 modest games + 2 huge outliers → mean ~17, median 12
    df = pd.DataFrame({
        "PLAYER_NAME": ["P"] * 12,
        "TEAM_ABBREVIATION": ["LVA"] * 12,
        "_date": pd.date_range("2026-07-01", periods=12),
        "PTS": [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 40, 40],
        "AST": [3] * 12, "REB": [4] * 12, "FG3M": [1] * 12,
    })
    # Median = 12 → line lands around 11.5 or 12.5, NOT the mean-driven ~17
    line = _synthetic_line_from_recent(df, "PTS")
    assert line is not None
    assert 11.0 <= line <= 13.0, f"expected median-anchored line ~12, got {line}"


def test_blend_projection_clamps_extreme_low():
    # Model produces absurdly low prediction (1.0 for a 4.5 avg player)
    blended = _blend_projection(model_pred=1.0, l20_avg=4.5, stat="REB")
    # Should clamp to >= 0.4 * 4.5 = 1.8
    assert blended >= 1.8


def test_blend_projection_clamps_extreme_high():
    blended = _blend_projection(model_pred=50.0, l20_avg=10.0, stat="REB")
    # Should clamp to <= 1.6 * 10 = 16.0
    assert blended <= 16.0


def test_blend_projection_returns_reasonable_middle():
    blended = _blend_projection(model_pred=18.0, l20_avg=20.0, stat="PTS")
    # 0.65 * 18 + 0.35 * 20 = 18.7
    assert 18.0 < blended < 19.5


def test_min_avg_min_filter_excludes_bench_players():
    df = pd.DataFrame({
        "PLAYER_NAME": ["Bench"] * 12,
        "TEAM_ABBREVIATION": ["LVA"] * 12,
        "_date": pd.date_range("2026-07-01", periods=12),
        "MIN": [4.5] * 12,  # bench player, way under 15
        "PTS": [10] * 12, "AST": [3] * 12, "REB": [6] * 12, "FG3M": [1] * 12,
    })
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    preds = {"PTS": _FakeModel(11.0), "AST": _FakeModel(3.0),
             "REB": _FakeModel(6.0), "FG3M": _FakeModel(1.0)}
    props = generate_wnba_props(
        df, _fake_getter(preds), odds={},
        todays_games=tonight, synthesize_missing=True, min_avg_min=15.0,
    )
    assert props == [], "bench player should be filtered out entirely"


def test_min_avg_min_filter_keeps_starter():
    df = pd.DataFrame({
        "PLAYER_NAME": ["Starter"] * 12,
        "TEAM_ABBREVIATION": ["LVA"] * 12,
        "_date": pd.date_range("2026-07-01", periods=12),
        "MIN": [28] * 12,
        "PTS": [18] * 12, "AST": [4] * 12, "REB": [8] * 12, "FG3M": [1] * 12,
    })
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    preds = {"PTS": _FakeModel(19.0), "AST": _FakeModel(4.0),
             "REB": _FakeModel(8.5), "FG3M": _FakeModel(1.0)}
    props = generate_wnba_props(
        df, _fake_getter(preds), odds={},
        todays_games=tonight, synthesize_missing=True, min_avg_min=15.0,
    )
    stats = {p.stat for p in props}
    assert "PTS" in stats
    assert "REB" in stats


def test_reb_ast_combo_stat_supported():
    df = pd.DataFrame({
        "PLAYER_NAME": ["Star"] * 12,
        "TEAM_ABBREVIATION": ["LVA"] * 12,
        "_date": pd.date_range("2026-07-01", periods=12),
        "MIN": [30] * 12,
        "PTS": [15] * 12, "AST": [4] * 12, "REB": [7] * 12, "FG3M": [1] * 12,
    })
    tonight = [{"home": {"abbrev": "LVA"}, "away": {"abbrev": "ATL"}}]
    preds = {"PTS": _FakeModel(16.0), "AST": _FakeModel(4.0), "REB": _FakeModel(7.5), "FG3M": _FakeModel(1.0)}
    props = generate_wnba_props(
        df, _fake_getter(preds), odds={},
        todays_games=tonight, synthesize_missing=True,
    )
    combo_stats = {p.stat for p in props if p.stat == "REB+AST"}
    assert "REB+AST" in combo_stats
    reb_ast_props = [p for p in props if p.stat == "REB+AST"]
    # REB+AST projection = 7.5 + 4 = 11.5 (blended with L20 avg 11 → similar)
    assert 9.0 <= reb_ast_props[0].projected <= 13.0
