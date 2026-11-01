from datetime import datetime, timezone

import polars as pl
import pytest

from cbbpred import config as cfg
from cbbpred import lines
from cbbpred.oddsmath import devig_two_way

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc)


def item(spread=-7.5, favorite=True, total=147.5, provider="DraftKings", hml=-325, aml=260):
    return {"provider": {"name": provider}, "spread": spread, "overUnder": total, "overOdds": -110.0, "underOdds": -110.0,
            "homeTeamOdds": {"favorite": favorite, "moneyLine": hml, "spreadOdds": -105.0,
                             "open": {"pointSpread": {"american": "-6.5"}}},
            "awayTeamOdds": {"favorite": not favorite, "moneyLine": aml, "spreadOdds": -115.0},
            "open": {"total": {"american": "148.5"}}}


def test_home_spread_flips_to_home_favored_positive():
    row = lines.parse_odds({"items": [item()]}, "1", NOW)[0]
    assert row["spread_line"] == 7.5 and row["spread_open"] == 6.5 and row["total_open"] == 148.5
    assert row["home_moneyline"] == -325 and row["provider"] == "DraftKings"


def test_home_underdog_gets_negative_spread_line():
    row = lines.parse_odds({"items": [item(spread=4.0, favorite=False, hml=150, aml=-180)]}, "1", NOW)[0]
    assert row["spread_line"] == -4.0


def test_spread_disagreeing_with_favorite_flag_is_dropped_but_total_kept():
    row = lines.parse_odds({"items": [item(spread=-7.5, favorite=False)]}, "1", NOW)[0]
    assert row["spread_line"] is None and row["total_line"] == 147.5


def test_empty_items_gives_no_rows():
    assert lines.parse_odds({"items": []}, "1", NOW) == []


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CBB_DATA_DIR", str(tmp_path))
    return tmp_path


def test_history_only_appends_when_a_line_moves(data_dir):
    row = lines.parse_odds({"items": [item()]}, "1", NOW)
    assert lines.append_history(row) == 1
    assert lines.append_history(lines.parse_odds({"items": [item()]}, "1", NOW)) == 0
    moved = lines.parse_odds({"items": [item(spread=-8.5)]}, "1", NOW)
    assert lines.append_history(moved) == 1
    assert pl.read_parquet(cfg.odds_history_path()).height == 2


def test_preferred_provider_wins_and_latest_value_is_used(data_dir):
    rows = lines.parse_odds({"items": [item(provider="consensus", spread=-3.0), item(provider="DraftKings", spread=-7.5)]}, "1", NOW)
    lines.append_history(rows)
    lines.append_history(lines.parse_odds({"items": [item(spread=-8.5)]}, "1", NOW.replace(hour=15)))
    best = lines.latest_lines().row(0, named=True)
    assert best["line_provider"] == "DraftKings" and best["spread_line"] == 8.5
    moves = lines.movement_index()["1"]
    assert [m["spread_home"] for m in moves] == [-7.5, -8.5]


def test_devig_removes_the_margin():
    a, b = devig_two_way(-110, -110)
    assert abs(a - 0.5) < 1e-9 and abs(a + b - 1) < 1e-9
    assert devig_two_way(None, -110) is None
