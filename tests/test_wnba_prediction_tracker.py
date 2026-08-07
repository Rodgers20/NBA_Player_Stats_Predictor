"""Tests for utils.wnba_prediction_tracker (uses tmp storage, no network)."""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from utils import wnba_prediction_tracker as tracker
from utils.wnba_props import WnbaProp


@pytest.fixture
def tmp_history(monkeypatch, tmp_path):
    """Redirect the history file to a temporary location per test."""
    p = tmp_path / "prediction_history.json"
    monkeypatch.setattr(tracker, "_HISTORY_FILE", p)
    yield p


def _prop(name, stat="PTS", pick="OVER", line=15.5, projected=17.5,
          conf="HIGH", hit_prob=0.7, ev=0.15):
    return WnbaProp(
        player_name=name, team="LVA", stat=stat, line=line, projected=projected,
        edge=projected - line, pick=pick, hit_prob=hit_prob, ev=ev,
        over_price=-110, under_price=-110, bookmaker="FanDuel", confidence=conf,
    )


def test_record_predictions_writes_new_date(tmp_history):
    n = tracker.record_predictions([_prop("A"), _prop("B")], target_date="2026-08-01")
    assert n == 2
    data = json.loads(tmp_history.read_text())
    assert "2026-08-01" in data
    assert len(data["2026-08-01"]) == 2


def test_record_predictions_noop_if_date_already_present(tmp_history):
    tracker.record_predictions([_prop("A")], target_date="2026-08-01")
    n = tracker.record_predictions([_prop("B")], target_date="2026-08-01")
    assert n == 0
    data = json.loads(tmp_history.read_text())
    assert len(data["2026-08-01"]) == 1
    assert data["2026-08-01"][0]["player_name"] == "A"


def test_grade_date_marks_win_loss_push_and_dnp(tmp_history):
    tracker.record_predictions([
        _prop("Winner",  pick="OVER",  line=15.5, projected=20),
        _prop("Loser",   pick="OVER",  line=15.5, projected=20),
        _prop("Pusher",  pick="OVER",  line=15.0, projected=20),
        _prop("Ghost",   pick="OVER",  line=15.5, projected=20),
    ], target_date="2026-08-02")

    game_log = pd.DataFrame({
        "PLAYER_NAME": ["Winner", "Loser", "Pusher"],
        "_date": pd.to_datetime(["2026-08-02"] * 3),
        "PTS": [20, 10, 15],
    })

    graded = tracker.grade_date("2026-08-02", game_log)
    assert graded == 4

    data = json.loads(tmp_history.read_text())
    outcomes = {e["player_name"]: e["outcome"] for e in data["2026-08-02"]}
    assert outcomes == {"Winner": "WIN", "Loser": "LOSS", "Pusher": "PUSH", "Ghost": "DNP"}


def test_grade_date_handles_combo_stats(tmp_history):
    tracker.record_predictions([
        _prop("Combo Star", stat="PTS+AST", pick="OVER", line=22.5, projected=25),
    ], target_date="2026-08-02")

    game_log = pd.DataFrame({
        "PLAYER_NAME": ["Combo Star"],
        "_date": pd.to_datetime(["2026-08-02"]),
        "PTS": [20], "AST": [5],   # sum = 25 > 22.5 → WIN
    })
    tracker.grade_date("2026-08-02", game_log)
    data = json.loads(tmp_history.read_text())
    entry = data["2026-08-02"][0]
    assert entry["actual"] == 25.0
    assert entry["outcome"] == "WIN"


def test_accuracy_summary_counts_correctly(tmp_history):
    tracker.record_predictions([
        _prop("W1", conf="HIGH", pick="OVER", line=10, projected=12),
        _prop("W2", conf="HIGH", pick="OVER", line=10, projected=12),
        _prop("L1", conf="MED",  pick="OVER", line=10, projected=12),
    ], target_date=date.today().isoformat())

    game_log = pd.DataFrame({
        "PLAYER_NAME": ["W1", "W2", "L1"],
        "_date": pd.to_datetime([date.today().isoformat()] * 3),
        "PTS": [15, 20, 5],
    })
    tracker.grade_date(date.today().isoformat(), game_log)

    summary = tracker.get_accuracy_summary(lookback_days=30)
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["hit_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert summary["high_wins"] == 2
    assert summary["high_losses"] == 0
    assert summary["high_hit_rate"] == 1.0
