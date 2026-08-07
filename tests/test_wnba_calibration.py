"""Tests for the calibration offsets computed by wnba_prediction_tracker."""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from utils import wnba_prediction_tracker as tracker
from utils.wnba_props import WnbaProp


@pytest.fixture
def tmp_history(monkeypatch, tmp_path):
    hist = tmp_path / "prediction_history.json"
    calib = tmp_path / "model_calibration.json"
    monkeypatch.setattr(tracker, "_HISTORY_FILE", hist)
    monkeypatch.setattr(tracker, "_CALIBRATION_FILE", calib)
    yield hist, calib


def _prop(name, stat="PTS", pick="OVER", line=15.5, projected=17.5):
    return WnbaProp(
        player_name=name, team="LVA", stat=stat, line=line, projected=projected,
        edge=projected - line, pick=pick, hit_prob=0.6, ev=0.10,
        over_price=-110, under_price=-110, bookmaker="FanDuel", confidence="MED",
    )


def test_calibration_requires_min_samples(tmp_history):
    hist_file, calib_file = tmp_history
    # Only 3 samples — below default min_samples=25
    tracker.record_predictions([_prop("A"), _prop("B"), _prop("C")], target_date="2026-07-30")
    log = pd.DataFrame({
        "PLAYER_NAME": ["A", "B", "C"],
        "_date": pd.to_datetime(["2026-07-30"] * 3),
        "PTS": [12, 14, 16],
    })
    tracker.grade_date("2026-07-30", log)
    offsets = tracker.compute_calibration_offsets(min_samples=25)
    assert offsets == {}
    assert not calib_file.exists()


def test_calibration_writes_offsets_when_enough_samples(tmp_history):
    hist_file, calib_file = tmp_history
    # 30 predictions, model consistently over-predicts by 5 points
    props = [_prop(f"P{i}", projected=20.0, line=15.5) for i in range(30)]
    tracker.record_predictions(props, target_date="2026-07-30")

    log = pd.DataFrame({
        "PLAYER_NAME": [f"P{i}" for i in range(30)],
        "_date": pd.to_datetime(["2026-07-30"] * 30),
        "PTS": [15] * 30,   # actual = 15, projected = 20 → offset = +5
    })
    tracker.grade_date("2026-07-30", log)

    offsets = tracker.compute_calibration_offsets(min_samples=10)
    assert "PTS" in offsets
    assert offsets["PTS"] == pytest.approx(5.0)

    # Persisted to disk
    stored = json.loads(calib_file.read_text())
    assert stored["PTS"] == pytest.approx(5.0)


def test_get_calibration_offsets_returns_empty_when_no_file(tmp_history):
    assert tracker.get_calibration_offsets() == {}


def test_calibration_offsets_by_stat(tmp_history):
    _, calib_file = tmp_history
    # 15 PTS predictions off by +3, 15 REB predictions off by -1
    pts_props = [_prop(f"PTS{i}", stat="PTS", projected=20.0, line=15.5) for i in range(15)]
    reb_props = [_prop(f"REB{i}", stat="REB", projected=8.0,  line=6.5) for i in range(15)]
    tracker.record_predictions(pts_props + reb_props, target_date="2026-07-29")

    log = pd.DataFrame({
        "PLAYER_NAME": [f"PTS{i}" for i in range(15)] + [f"REB{i}" for i in range(15)],
        "_date": pd.to_datetime(["2026-07-29"] * 30),
        "PTS": [17] * 15 + [0] * 15,
        "REB": [0] * 15 + [9] * 15,
    })
    tracker.grade_date("2026-07-29", log)

    offsets = tracker.compute_calibration_offsets(min_samples=10)
    assert offsets["PTS"] == pytest.approx(3.0)   # over by 3
    assert offsets["REB"] == pytest.approx(-1.0)  # under by 1
