from futball.props import forms


def row(pid, team, event, date, *, started=True, goals=0, assists=0, sot=0, shots=0, pos="F", comp="E0", team_goals=1, opp_goals=1, sub_in=False):
    return {"espn_id": pid, "player": f"P{pid}", "team": team, "opponent": "X", "competition_code": comp, "date": date,
            "event_id": event, "is_home": True, "started": started, "sub_in": sub_in, "position": pos, "goals": goals,
            "assists": assists, "shots": shots, "sot": sot, "fouls_committed": 0, "fouls_suffered": 0, "yellow": 0,
            "red": 0, "offsides": 0, "saves": 0, "team_goals": team_goals, "opp_goals": opp_goals}


# Arsenal play 6 matches (e1..e6). P1 starts all, P2 is a sub in e1-e3 then misses the rest, P3 starts only e5, e6.
ROWS = (
    [row("1", "Arsenal", f"e{i}", f"2026-10-{i:02d}", goals=1 if i % 2 else 0, sot=2, shots=3) for i in range(1, 7)]
    + [row("2", "Arsenal", f"e{i}", f"2026-10-{i:02d}", started=False, sub_in=True, sot=1) for i in range(1, 4)]
    + [row("3", "Arsenal", f"e{i}", f"2026-10-{i:02d}", pos="M", assists=1) for i in (5, 6)]
)


def by_id(rows, before=None):
    return {f.espn_id: f for f in forms.build_forms(rows, before=before)}


def test_counts_games_starts_subs_and_team_games():
    f = by_id(ROWS)
    assert (f["1"].games, f["1"].starts, f["1"].subs, f["1"].team_games) == (6, 6, 0, 6)
    assert (f["2"].games, f["2"].starts, f["2"].subs) == (3, 0, 3)
    assert (f["1"].goals, f["1"].sot, f["1"].shots) == (3, 12, 18)
    assert f["3"].assists == 2 and f["3"].position == "M"


def test_last_five_window_and_gap_since_last_appearance():
    f = by_id(ROWS)
    assert (f["1"].last5_starts, f["1"].last5_games, f["1"].gap) == (5, 5, 0)
    assert (f["2"].last5_starts, f["2"].gap) == (0, 3)  # last seen e3; e4, e5, e6 came and went
    assert (f["3"].last5_starts, f["3"].gap) == (2, 0)


def test_before_cutoff_only_uses_earlier_matches():
    f = by_id(ROWS, before="2026-10-04")  # e1..e3
    assert f["1"].team_games == 3 and f["1"].games == 3
    assert "3" not in f  # had not played yet


def test_player_belongs_to_the_team_of_his_latest_appearance():
    moved = ROWS + [row("2", "Chelsea", "c1", "2026-10-20", started=True), row("9", "Chelsea", "c1", "2026-10-20")]
    f = by_id(moved)
    assert f["2"].team == "Chelsea" and f["2"].games == 1


def test_competition_is_the_domestic_league_not_the_cup():
    rows = [row("1", "Arsenal", "e1", "2026-10-01", comp="E0"), row("1", "Arsenal", "u1", "2026-10-08", comp="UCL")]
    assert by_id(rows)["1"].competition_code == "E0"


def test_team_aggregates_support_the_team_factor():
    agg = forms.team_aggregates(ROWS)
    arsenal = agg["Arsenal"]
    assert arsenal.games == 6
    assert arsenal.goals_for == 6 and arsenal.sot_for == 6 * 2 + 3 * 1  # per-event team totals use player sums
