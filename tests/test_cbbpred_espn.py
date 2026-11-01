from datetime import datetime, timezone

from cbbpred import espn
from tests.cbb_helpers import A, B, event


def test_parse_final_game_reads_scores_rank_and_overtimes():
    row = espn.parse_event(event("9", "2026-11-04T23:30Z", A, B, 81, 77, rank=7, period=3))
    assert row["status"] == "final" and (row["home_score"], row["away_score"]) == (81, 77)
    assert row["home_rank"] == 7 and row["overtimes"] == 1
    assert row["gameday"] == "2026-11-04" and row["gametime"] == "18:30"  # Eastern time


def test_unranked_99_becomes_none_and_scheduled_has_no_score():
    row = espn.parse_event(event("9", "2026-11-04T23:30Z", A, B, state="pre", name="STATUS_SCHEDULED"))
    assert row["home_rank"] is None and row["home_score"] is None and row["status"] == "scheduled"


def test_postponed_and_live_statuses():
    assert espn.parse_event(event("1", "2026-11-04T23:30Z", A, B, state="post", name="STATUS_POSTPONED"))["status"] == "postponed"
    assert espn.parse_event(event("2", "2026-11-04T23:30Z", A, B, state="in", name="STATUS_HALFTIME"))["status"] == "live"


def test_preseason_and_tbd_events_are_skipped():
    ev = event("1", "2026-11-04T23:30Z", A, B)
    ev["season"]["type"] = 1
    assert espn.parse_event(ev) is None
    assert espn.parse_event(event("2", "2026-11-04T23:30Z", ("-1", "TBD", None), B)) is None


def test_missing_conference_is_none():
    assert espn.parse_event(event("1", "2026-11-04T23:30Z", ("5", "DII", None), B))["home_conf"] is None


def test_current_season_flips_in_summer():
    assert espn.current_season(datetime(2026, 11, 1, tzinfo=timezone.utc)) == 2027
    assert espn.current_season(datetime(2027, 3, 1, tzinfo=timezone.utc)) == 2027
    assert espn.current_season(datetime(2027, 7, 1, tzinfo=timezone.utc)) == 2028
