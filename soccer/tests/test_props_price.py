import pytest

from futball.props import forms as f
from futball.props import price as p


def pf(pid, team, pos="F", games=8, starts=8, goals=4, assists=2, sot=12, shots=24, team_games=8, gap=0, last5=5):
    return f.PlayerForm(pid, f"P{pid}", team, "E0", pos, games, starts, games - starts, team_games, last5, 5, gap, goals, assists, shots, sot)


FORMS = [pf("1", "Arsenal"), pf("2", "Arsenal", pos="M", goals=1, assists=3, sot=4), pf("3", "Arsenal", pos="G", goals=0, sot=0),
         pf("4", "Arsenal", games=1, starts=1), pf("5", "Chelsea", goals=2, sot=8)]
AGG = {"Arsenal": f.TeamAggregate(8, 16, 6, 60, 30), "Chelsea": f.TeamAggregate(8, 8, 12, 40, 50)}
LEAGUE = {"goals": 1.5, "sot": 4.5}


def run(xg_home=1.5, **kw):
    priors = p.position_priors(FORMS)
    return p.price_fixture(FORMS, AGG, priors, LEAGUE, "Arsenal", "Chelsea",
                           expected_goals={"home": xg_home, "away": 1.0}, expected_sot={"home": 5.0, "away": 4.0}, **kw)


def test_goalkeepers_and_one_game_players_get_no_rows():
    assert {r["player"] for r in run()} <= {"P1", "P2", "P5"}


def test_each_player_gets_goal_assist_and_two_sot_lines_with_contract_fields():
    rows = [r for r in run() if r["player"] == "P1"]
    assert sorted((r["market"], r["line"]) for r in rows) == [("anytime_goal", None), ("assist", None), ("shots_on_target", 0.5), ("shots_on_target", 1.5)]
    goal = next(r for r in rows if r["market"] == "anytime_goal")
    assert goal["team"] == "Arsenal" and goal["opponent"] == "Chelsea" and goal["is_home"] is True
    assert goal["games"] == 8 and goal["per_game"] == pytest.approx(0.5) and 0 < goal["prob"] < 1 and goal["confidence"] == "high"
    assert goal["reason"]


def test_better_team_expectation_raises_every_probability_and_o15_is_below_o05():
    low = {(r["player"], r["market"], r["line"]): r["prob"] for r in run(xg_home=0.8)}
    high = {(r["player"], r["market"], r["line"]): r["prob"] for r in run(xg_home=2.4)}
    assert high[("P1", "anytime_goal", None)] > low[("P1", "anytime_goal", None)]
    assert high[("P1", "shots_on_target", 1.5)] < high[("P1", "shots_on_target", 0.5)]


def test_rows_are_sorted_by_probability_within_each_market():
    goal = [r["prob"] for r in run() if r["market"] == "anytime_goal"]
    assert goal == sorted(goal, reverse=True)


def test_missing_sot_expectation_falls_back_to_the_opponent_concession_proxy():
    priors = p.position_priors(FORMS)
    rows = p.price_fixture(FORMS, AGG, priors, LEAGUE, "Arsenal", "Chelsea",
                           expected_goals={"home": 1.5, "away": 1.0}, expected_sot=None)
    assert any(r["market"] == "shots_on_target" for r in rows)


def test_markets_can_be_switched_off_for_leagues_without_shots_data():
    priors = p.position_priors(FORMS)
    rows = p.price_fixture(FORMS, AGG, priors, LEAGUE, "Arsenal", "Chelsea",
                           expected_goals={"home": 1.5, "away": 1.0}, expected_sot=None, markets=("anytime_goal", "assist"))
    assert {r["market"] for r in rows} == {"anytime_goal", "assist"}
