"""Contract checks against the exported files (skipped when no export has been generated yet)."""

import json
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "data" / "nhl"
pytestmark = pytest.mark.skipif(not (OUT / "health.json").exists(), reason="no NHL export on disk")


def load(name):
    return json.loads((OUT / name).read_text())


def test_health_and_index():
    h = load("health.json")
    assert {"generated_at", "season", "counts", "model", "schedule_days", "lineup_note"} <= set(h)
    idx = load("schedule/index.json")
    assert idx["days"] and all((OUT / "schedule" / f"{d['date']}.json").exists() for d in idx["days"])


def test_every_game_has_its_matchup_file_and_probabilities_are_valid():
    for d in load("schedule/index.json")["days"]:
        for g in load(f"schedule/{d['date']}.json")["games"]:
            assert (OUT / "game" / f"{g['id']}.json").exists()
            if g["pick"]:
                assert abs(g["pick"]["p_home"] + g["pick"]["p_away"] - 1) < 1e-3
                assert g["pick"]["confidence"] in ("low", "medium", "high")


def test_props_board_shape():
    b = load("props.json")
    assert b["players"], "board should not be empty while games are scheduled"
    for row in b["players"][:50]:
        assert set(row["markets"]) == {"goal", "sog", "assists", "points"}
        for m in row["markets"].values():
            for ln in m["lines"]:
                assert 0 < ln["p_over"] < 1 and abs(ln["p_over"] + ln["p_under"] - 1) < 1e-3
        assert [x["line"] for x in row["markets"]["goal"]["lines"]] == [0.5]
    # sog lines are monotone: a higher line is never more likely
    for row in b["players"][:50]:
        ps = [x["p_over"] for x in row["markets"]["sog"]["lines"]]
        assert ps == sorted(ps, reverse=True)


def test_player_pages_exist_for_every_indexed_player():
    for p in load("players.json")["players"][:25]:
        page = load(f"player/{p['id']}.json")
        assert page["player"]["id"] == p["id"] and "hit_rates" in page and "log" in page
