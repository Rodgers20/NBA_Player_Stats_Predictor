from datetime import date

import polars as pl

from futball.props import service


IDENTITIES = {
    "bukayo saka": {"name": "Bukayo Saka", "event_ids": [], "league": [{"player_id": 111, "team": "Arsenal FC", "competition_code": "E0"}]},
    "kylian mbappe": {"name": "Kylian Mbappé", "event_ids": [3412], "league": [{"player_id": 222, "team": "Real Madrid CF", "competition_code": "SP1"}]},
    "euro only": {"name": "Euro Only", "event_ids": [9], "league": []},
}


def lake(name, code):
    return {"Arsenal FC": "Arsenal", "Real Madrid CF": "Real Madrid"}.get(name)


def test_key_resolver_matches_name_and_club_to_the_public_profile_id_else_espn():
    key_for = service.make_key_resolver(IDENTITIES, lake_name=lake, public_id=lambda e: str(e["event_ids"][0]) if e["event_ids"] else f"fd-{e['league'][0]['player_id']}")
    assert key_for("77", "Bukayo Saka", "Arsenal") == "fd-111"
    assert key_for("78", "Kylian Mbappé", "Real Madrid") == "3412"
    assert key_for("79", "Bukayo Saka", "Chelsea") == "espn-79"   # same name, other club: not the same person
    assert key_for("80", "Somebody New", "Arsenal") == "espn-80"


def test_upcoming_window_keeps_only_scheduled_fixtures_inside_the_days():
    df = pl.DataFrame({
        "competition_code": ["E0"] * 5 + ["SP1"], "date": ["2026-11-02", "2026-11-03", "2026-11-10", "2026-11-17", "2026-11-18", "2026-11-04"],
        "time": ["12:00"] * 6, "kickoff_utc": ["x"] * 6, "home_team": list("ABCDEF"), "away_team": list("UVWXYZ"),
        "status": ["finished", "scheduled", "scheduled", "scheduled", "scheduled", "scheduled"],
        "fixture_key": [f"k{i}" for i in range(6)],
    })
    got = service.upcoming_fixtures(df, "E0", today=date(2026, 11, 3), days=14)
    assert [f["fixture_key"] for f in got] == ["k1", "k2", "k3"]  # not the finished one, not day 15, not another league
