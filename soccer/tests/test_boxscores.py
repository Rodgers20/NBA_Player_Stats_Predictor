from datetime import date

from futball import boxscores as bs


def player(pid, name, *, starter, appeared=1.0, **stats):
    base = {"appearances": appeared, "totalGoals": 0.0, "goalAssists": 0.0, "totalShots": 0.0,
            "shotsOnTarget": 0.0, "foulsCommitted": 0.0, "foulsSuffered": 0.0, "yellowCards": 0.0,
            "redCards": 0.0, "offsides": 0.0, "saves": 0.0, "subIns": 0.0}
    base.update(stats)
    return {"athlete": {"id": pid, "displayName": name}, "starter": starter, "subbedIn": not starter and appeared > 0,
            "position": {"abbreviation": "F"}, "stats": [{"name": k, "value": v} for k, v in base.items()]}


BODY = {"rosters": [
    {"homeAway": "home", "team": {"id": "1", "displayName": "Arsenal"}, "roster": [
        player("10", "Bukayo Saka", starter=True, totalGoals=1.0, totalShots=4.0, shotsOnTarget=2.0),
        player("11", "Leandro Trossard", starter=False, appeared=1.0, totalShots=1.0, subIns=1.0),
        player("12", "Unused Sub", starter=False, appeared=0.0),
    ]},
    {"homeAway": "away", "team": {"id": "2", "displayName": "Chelsea"}, "roster": [
        player("20", "Cole Palmer", starter=True, goalAssists=1.0, yellowCards=1.0),
    ]},
]}
META = {"event_id": "999", "date": "2026-11-01", "home": "Arsenal", "away": "Chelsea", "home_goals": 2, "away_goals": 1}


def rows():
    return bs.parse_summary(BODY, "E0", META, resolve=lambda name, code: name)


def test_only_players_who_appeared_become_rows():
    assert [r["player"] for r in rows()] == ["Bukayo Saka", "Leandro Trossard", "Cole Palmer"]


def test_row_carries_match_context_and_stats():
    saka = rows()[0]
    assert (saka["team"], saka["opponent"], saka["is_home"]) == ("Arsenal", "Chelsea", True)
    assert (saka["goals"], saka["shots"], saka["sot"], saka["started"]) == (1, 4, 2, True)
    assert (saka["team_goals"], saka["opp_goals"], saka["date"], saka["espn_id"]) == (2, 1, "2026-11-01", "10")


def test_substitute_is_flagged_and_away_side_is_not_home():
    trossard, palmer = rows()[1], rows()[2]
    assert trossard["started"] is False and trossard["sub_in"] is True
    assert (palmer["is_home"], palmer["team_goals"], palmer["assists"], palmer["yellow"]) == (False, 1, 1, 1)


def test_missing_stat_is_zero_not_none_but_missing_rosters_yield_nothing():
    assert bs.parse_summary({}, "E0", META, resolve=lambda n, c: n) == []
    body = {"rosters": [{"homeAway": "home", "team": {"displayName": "Arsenal"},
                         "roster": [{"athlete": {"id": "1", "displayName": "X"}, "starter": True, "stats": [{"name": "appearances", "value": 1}]}]}]}
    assert bs.parse_summary(body, "E0", META, resolve=lambda n, c: n)[0]["shots"] == 0


def test_completed_events_reads_scoreboard():
    board = {"events": [
        {"id": "1", "date": "2026-11-01T15:00Z", "status": {"type": {"completed": True}}, "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"displayName": "A"}, "score": "2"}, {"homeAway": "away", "team": {"displayName": "B"}, "score": "0"}]}]},
        {"id": "2", "date": "2026-11-01T17:00Z", "status": {"type": {"completed": False}}, "competitions": [{"competitors": []}]},
    ]}
    got = bs.completed_events(board)
    assert [(e["event_id"], e["home"], e["away"], e["home_goals"], e["away_goals"]) for e in got] == [("1", "A", "B", 2, 0)]


def test_all_star_exhibitions_are_not_league_matches():
    def ev(eid, home, away):
        return {"id": eid, "date": "2026-07-29T01:00Z", "status": {"type": {"completed": True}}, "competitions": [{"competitors": [
            {"homeAway": "home", "team": {"displayName": home}, "score": "1"}, {"homeAway": "away", "team": {"displayName": away}, "score": "1"}]}]}
    got = bs.completed_events({"events": [ev("1", "MLS All-Stars", "Liga MX All-Stars"), ev("2", "Austin FC", "FC Dallas")]})
    assert [e["event_id"] for e in got] == ["2"]


def test_season_start_is_august_for_europe_and_january_for_mls():
    assert bs.season_start("E0", today=date(2026, 11, 3)) == date(2026, 8, 1)
    assert bs.season_start("E0", today=date(2027, 2, 3)) == date(2026, 8, 1)
    assert bs.season_start("MLS", today=date(2026, 11, 3)) == date(2026, 1, 1)


def test_leaders_rank_goals_and_assists_from_box_scores_for_one_competition():
    import polars as pl

    def r(pid, name, team, event, goals, assists, comp="MLS"):
        return {"espn_id": pid, "player": name, "team": team, "competition_code": comp, "event_id": event, "goals": goals, "assists": assists}
    df = pl.DataFrame([
        r("1", "A", "X", "e1", 2, 0), r("1", "A", "X", "e2", 1, 1), r("2", "B", "Y", "e1", 1, 3), r("2", "B", "Y", "e2", 0, 1),
        r("3", "C", "Z", "e1", 9, 9, comp="E0"),
    ])
    got = bs.leaders("MLS", limit=5, frame=df)
    assert [(x["player"], x["goals"], x["played"]) for x in got["scorers"]] == [("A", 3, 2), ("B", 1, 2)]
    assert [(x["player"], x["assists"]) for x in got["assists"]] == [("B", 4), ("A", 1)]
    assert got["scorers"][0]["player_id"] == "espn-1" and got["scorers"][0]["team"] == "X"
