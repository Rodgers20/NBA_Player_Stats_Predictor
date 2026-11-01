import json
from datetime import datetime, timezone

import pytest

from cbbpred import config as cfg
from cbbpred import export
from cbbpred.store import write_json, write_parquet
from tests.cbb_helpers import CONFS, season_games

NOW = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setenv("CBB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CBB_MODEL_DIR", str(tmp_path / "models"))
    write_parquet(season_games(), cfg.games_path())
    write_json(CONFS, cfg.conferences_path())
    out = tmp_path / "out"
    summary = export.export_all(NOW, out)
    return out, summary


def load(out, rel):
    return json.loads((out / rel).read_text())


def test_all_top_level_files_exist_and_parse(exported):
    out, _ = exported
    for name in ("health.json", "rankings.json", "standings.json", "model-performance.json"):
        assert isinstance(load(out, name), dict)


def test_schedule_covers_the_window_and_has_game_rows(exported):
    out, summary = exported
    assert summary["schedule_days"] == cfg.EXPORT_DAYS_BACK + cfg.EXPORT_DAYS_AHEAD + 1
    day = load(out, "schedule/2026-11-04.json")
    game = day["games"][0]
    assert {"id", "home", "away", "line", "pick", "status", "score", "kickoff"} <= set(game)
    assert {"id", "name", "abbr", "logo", "record", "rank"} <= set(game["home"])
    assert game["status"] == "final" and game["score"]["home"] == 75


def test_game_file_has_form_h2h_and_ratings_blocks(exported):
    out, _ = exported
    g = load(out, "game/1005.json")
    for side in ("home", "away"):
        assert set(g["form"][side]) == {"overall", "split"}
        assert {"games", "record", "avg_scored", "avg_allowed", "avg_margin"} <= set(g["form"][side]["overall"])
    assert "n" in g["h2h"] and "line_movement" in g and "why" in g


def test_form_never_includes_the_game_itself_or_later_games(exported):
    out, _ = exported
    g = load(out, "game/1005.json")
    kickoff_date = g["date"]
    assert all(e["date"] <= kickoff_date for e in g["form"]["home"]["overall"]["games"])


def test_stale_game_files_are_pruned(exported, tmp_path):
    out, _ = exported
    (out / "game" / "stale.json").write_text("{}")
    export.export_all(NOW, out)
    assert not (out / "game" / "stale.json").exists()


def test_files_are_compact_and_nan_free(exported):
    out, _ = exported
    text = (out / "rankings.json").read_text()
    assert "NaN" not in text and ": " not in text


def test_standings_group_teams_by_conference(exported):
    out, _ = exported
    s = load(out, "standings.json")
    acc = next(c for c in s["conferences"] if c["short"] == "ACC")
    assert {t["id"] for t in acc["teams"]} == {"1", "2"}
    assert all(t["conf_record"] and t["overall"] for t in acc["teams"])
