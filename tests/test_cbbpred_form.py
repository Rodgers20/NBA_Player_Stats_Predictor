from datetime import datetime, timezone

from cbbpred.form import TeamLogs
from tests.cbb_helpers import season_games

LATE = datetime(2027, 1, 1, tzinfo=timezone.utc)


def logs():
    g = season_games()
    return TeamLogs(g.filter(g["status"] == "final")), g


def test_form_is_oldest_to_newest_and_capped_at_ten():
    lg, _ = logs()
    form = lg.form("1", LATE)
    dates = [e["date"] for e in form["games"]]
    assert dates == sorted(dates) and form["n"] <= 10
    assert form["wins"] + form["losses"] == form["n"] and form["record"] == f"{form['wins']}-{form['losses']}"
    assert set(form["games"][0]) >= {"date", "opp", "opp_abbr", "home", "win", "pts_for", "pts_against"}


def test_split_keeps_only_home_or_only_road_games():
    lg, _ = logs()
    assert all(e["home"] for e in lg.form("1", LATE, at_home=True)["games"])
    assert not any(e["home"] for e in lg.form("1", LATE, at_home=False)["games"])


def test_lookups_are_strictly_before_tipoff():
    lg, g = logs()
    first = g.sort("kickoff").row(0, named=True)
    assert lg.form(first["home_id"], first["kickoff"])["n"] == 0
    assert lg.record(first["home_id"], first["kickoff"], first["season"]) == "0-0"


def test_averages_and_streak():
    lg, _ = logs()
    form = lg.form("1", LATE)
    pf = sum(e["pts_for"] for e in form["games"]) / form["n"]
    assert form["avg_scored"] == round(pf, 1)
    assert lg.streak("1", LATE, 2027) is not None


def test_head_to_head_is_from_the_home_teams_view():
    lg, _ = logs()
    h = lg.head_to_head("1", "2", LATE)
    assert h["n"] == h["home_wins"] + h["away_wins"] and h["n"] >= 1
    assert all(m["winner"] in ("home", "away") for m in h["games"])
