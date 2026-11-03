from datetime import date
from unittest.mock import patch

from futball.sources import espn_fixtures as ef


def event(eid, when, home, away, *, completed=False, hs="0", as_="0"):
    return {"id": eid, "date": when, "status": {"type": {"completed": completed, "state": "post" if completed else "pre"}},
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": home}, "score": hs},
                {"homeAway": "away", "team": {"displayName": away}, "score": as_}]}]}


def test_scheduled_event_has_no_score_and_a_kickoff_time():
    rec = ef.parse({"events": [event("1", "2026-11-08T00:30Z", "Inter Miami CF", "LA Galaxy")]}, "MLS")[0]
    assert (rec["date"], rec["time"], rec["home_team"], rec["away_team"]) == ("2026-11-08", "00:30", "Inter Miami CF", "LA Galaxy")
    assert rec["home_goals"] is None and rec["away_goals"] is None and rec["competition_code"] == "MLS"


def test_finished_event_keeps_its_score_including_zero():
    rec = ef.parse({"events": [event("2", "2026-11-01T20:00Z", "A", "B", completed=True, hs="0", as_="3")]}, "MLS")[0]
    assert (rec["home_goals"], rec["away_goals"]) == (0, 3)


def test_read_asks_one_date_at_a_time_and_survives_a_failed_day():
    calls = []

    def fake(code, day, today=None):
        calls.append(day)
        if day == date(2026, 11, 4):
            raise RuntimeError("boom")
        return {"events": [event(f"e{day.day}", f"{day:%Y-%m-%d}T20:00Z", "A", "B")]}

    with patch.object(ef.boxscores, "fetch_scoreboard", side_effect=fake):
        recs = ef.read("MLS", today=date(2026, 11, 3), days=2)
    assert [d.isoformat() for d in calls] == ["2026-11-03", "2026-11-04", "2026-11-05"]
    assert [r["date"] for r in recs] == ["2026-11-03", "2026-11-05"]
