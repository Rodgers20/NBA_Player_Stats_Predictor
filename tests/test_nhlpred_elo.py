import pandas as pd

from nhlpred.elo import EloParams, run_elo


def _games():
    return pd.DataFrame([
        {"game_id": 1, "season": 2025, "home_id": 1, "away_id": 2, "home_score": 3, "away_score": 1},
        {"game_id": 2, "season": 2025, "home_id": 1, "away_id": 2, "home_score": 2, "away_score": 3},
        {"game_id": 3, "season": 2025, "home_id": 2, "away_id": 1, "home_score": None, "away_score": None},
    ])


def test_no_leakage_and_updates():
    e, final = run_elo(_games())
    assert e.elo_h[0] == 1500 and e.elo_a[0] == 1500           # first game sees only the prior
    assert e.elo_h[1] > 1500 > e.elo_a[1]                       # winner of game 1 rated higher before game 2
    assert abs(sum(final.values()) - 3000) < 1e-6               # zero-sum
    # future game takes the ratings after game 2 and changes nothing
    assert e.elo_h[2] == final[2] and e.elo_a[2] == final[1]


def test_future_score_cannot_change_past():
    g = _games()
    g.loc[2, ["home_score", "away_score"]] = [9, 0]
    a, _ = run_elo(_games())
    b, _ = run_elo(g)
    assert list(a.elo_h[:3]) == list(b.elo_h[:3])


def test_season_regression():
    g = _games()
    g.loc[2, "season"] = 2026
    plain, _ = run_elo(g, EloParams(regress=0.0))
    reg, _ = run_elo(g, EloParams(regress=0.5))
    assert abs(reg.elo_h[2] - 1500) < abs(plain.elo_h[2] - 1500)
