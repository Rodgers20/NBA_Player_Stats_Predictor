from futball.props import board


def row(pid, team, opp, event, date, *, started=True, goals=0, assists=0, sot=0, shots=0, pos="F", comp="E0", home=True, tg=1, og=0):
    return {"espn_id": pid, "player": f"P{pid}", "team": team, "opponent": opp, "competition_code": comp, "date": date,
            "event_id": event, "is_home": home, "started": started, "sub_in": False, "position": pos, "goals": goals,
            "assists": assists, "shots": shots, "sot": sot, "fouls_committed": 1, "fouls_suffered": 2, "yellow": 0, "red": 0,
            "offsides": 0, "saves": 0, "team_goals": tg, "opp_goals": og}


def history():
    rows = []
    for i in range(1, 7):  # six matches each for Arsenal (home) and Chelsea, four players
        d = f"2026-10-{i:02d}"
        rows += [row("1", "Arsenal", "Chelsea", f"e{i}", d, goals=i % 2, sot=2, shots=3, tg=2, og=1),
                 row("2", "Arsenal", "Chelsea", f"e{i}", d, pos="M", assists=i % 3 == 0, sot=1, tg=2, og=1),
                 row("9", "Arsenal", "Chelsea", f"e{i}", d, pos="G", tg=2, og=1),
                 row("5", "Chelsea", "Arsenal", f"e{i}", d, home=False, goals=i % 3 == 0, sot=1, shots=2, tg=1, og=2)]
    return rows


FIXTURE = {"fixture_key": "E0|2026-11-07|Arsenal|Chelsea", "date": "2026-11-07", "time": "12:30",
           "kickoff_utc": "2026-11-07T12:30:00+00:00", "home_team": "Arsenal", "away_team": "Chelsea"}


def fake_predict(home, away):
    return {"expected_goals": {"home": 1.8, "away": 1.1},
            "sot": {"available": True, "home": {"expected": 5.5}, "away": {"expected": 4.0}}}


def build(code="E0", predict=fake_predict, fixtures=(FIXTURE,)):
    return board.build_board(code, history(), list(fixtures), predict, key_for=lambda eid, name, team: f"espn-{eid}",
                             generated_at="2026-11-03T18:00:00Z", trained_through="2026-11-02")


def test_board_follows_the_contract_shape():
    b = build()
    assert (b["competition_code"], b["generated_at"], b["model_trained_through"]) == ("E0", "2026-11-03T18:00:00Z", "2026-11-02")
    fx = b["fixtures"][0]
    assert fx["fixture_key"] == FIXTURE["fixture_key"] and fx["home_xg"] == 1.8 and fx["away_xg"] == 1.1
    prop = fx["props"][0]
    assert set(prop) >= {"player_key", "player", "team", "opponent", "is_home", "position", "market", "line", "prob",
                         "expected", "per_game", "games", "start_prob", "confidence", "reason"}
    assert "espn_id" not in prop and prop["player_key"].startswith("espn-")


def test_goalkeepers_are_excluded_and_markets_are_present():
    props = build()["fixtures"][0]["props"]
    assert "P9" not in {p["player"] for p in props}
    assert {(p["market"], p["line"]) for p in props} >= {("anytime_goal", None), ("shots_on_target", 0.5)}


def test_mls_has_no_shots_on_target_props():
    props = build(code="MLS")["fixtures"][0]["props"]
    assert {p["market"] for p in props} <= {"anytime_goal", "assist"}


def test_a_fixture_the_model_cannot_price_is_listed_without_props_instead_of_failing_the_board():
    def broken(home, away):
        raise ValueError("unknown team")
    fx = build(predict=broken)["fixtures"][0]
    assert fx["props"] == [] and fx["home_xg"] is None


def test_unavailable_sot_model_falls_back_to_the_proxy_and_still_prices_sot():
    def no_sot(home, away):
        return {"expected_goals": {"home": 1.8, "away": 1.1}, "sot": {"available": False}}
    props = build(predict=no_sot)["fixtures"][0]["props"]
    assert any(p["market"] == "shots_on_target" for p in props)


def test_gamelog_has_season_totals_newest_first_games_and_next_fixture():
    b = build()
    log = board.gamelog(history(), "1", "espn-1", b)
    assert log["player_key"] == "espn-1" and log["team"] == "Arsenal" and log["competition_code"] == "E0"
    assert log["season"]["games"] == 6 and log["season"]["starts"] == 6
    assert log["season"]["totals"]["sot"] == 12 and log["season"]["per_game"]["sot"] == 2.0
    assert [g["date"] for g in log["games"]] == sorted((g["date"] for g in log["games"]), reverse=True)
    assert log["games"][0]["result"] == "W 2-1" and log["games"][0]["opponent"] == "Chelsea"
    assert log["next_fixture"] == {"fixture_key": FIXTURE["fixture_key"], "opponent": "Chelsea", "date": "2026-11-07", "is_home": True}
    assert log["props"] and all(p["player_key"] == "espn-1" for p in log["props"])
    assert "tackles" in " ".join(log["notes"]).lower()
