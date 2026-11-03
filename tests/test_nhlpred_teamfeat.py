import numpy as np
import pandas as pd

from nhlpred.teamfeat import add_rolling, goalie_pre_svp, team_games


def _games(n=12):
    rows = []
    for i in range(n):
        played = i < n - 1
        rows.append({"game_id": i, "date": f"2026-11-{i + 1:02d}", "start_utc": f"2026-11-{i + 1:02d}T00:00Z", "season": 2027,
                     "home_id": 1 if i % 2 == 0 else 2, "away_id": 2 if i % 2 == 0 else 1,
                     "home_score": 5 if played else np.nan, "away_score": 1 if played else np.nan,
                     "home_sog": 30 if played else np.nan, "away_sog": 20 if played else np.nan})
    return pd.DataFrame(rows)


def test_rolling_is_pre_game_only():
    base = _games()
    a = add_rolling(team_games(base))
    changed = base.copy()
    changed.loc[changed.game_id == 8, ["home_score", "away_score"]] = [0, 9]   # change one result
    b = add_rolling(team_games(changed))
    early = a.game_id <= 8                                                   # rows up to and including game 8 see no change
    assert np.allclose(a.loc[early, "gf10"], b.loc[early, "gf10"])
    assert not np.allclose(a.loc[~early, "gf10"], b.loc[~early, "gf10"])


def test_first_game_is_the_prior_and_rest_flags():
    t = add_rolling(team_games(_games()))
    first = t[(t.team_id == 1)].iloc[0]
    assert abs(first.gf10 - 3.05) < 1e-9 and first.rest == 7
    assert (t[t.team_id == 1].rest.iloc[1], t[t.team_id == 1].b2b.iloc[1]) == (1, 1)   # games on consecutive days


def test_goalie_svp_excludes_current_game():
    g = pd.DataFrame({"player_id": [9, 9], "game_id": [1, 2], "date": ["2026-11-01", "2026-11-02"],
                      "saves": [0, 30], "shots_against": [30, 30]})
    out = goalie_pre_svp(g).sort_values("game_id")
    assert abs(out.svp_pre.iloc[0] - 0.905) < 1e-9          # first start: just the prior
    assert out.svp_pre.iloc[1] < 0.905                       # second start sees the first (awful) game only
