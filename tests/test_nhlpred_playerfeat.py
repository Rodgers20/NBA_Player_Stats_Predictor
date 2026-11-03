import numpy as np
import pandas as pd

from nhlpred import playerfeat as pf
from nhlpred.teamfeat import team_panel


def _world(n=30, upcoming=1):
    games, rows = [], []
    for i in range(n):
        played = i < n - upcoming
        d = f"2026-11-{i + 1:02d}" if i < 28 else f"2026-12-{i - 27:02d}"
        games.append({"game_id": i, "date": d, "start_utc": d + "T00:00Z", "season": 2027, "status": "final" if played else "scheduled",
                      "home_id": 1, "away_id": 2, "home_abbr": "AAA", "away_abbr": "BBB",
                      "home_score": 3 if played else np.nan, "away_score": 2 if played else np.nan,
                      "home_sog": 30 if played else np.nan, "away_sog": 28 if played else np.nan})
        if played:
            for pid, team, opp, home in ((10, 1, "BBB", True), (11, 1, "BBB", True), (20, 2, "AAA", False)):
                rows.append({"game_id": i, "date": d, "season": 2027, "player_id": pid, "name": f"P{pid}", "pos": "C", "team": "AAA" if team == 1 else "BBB",
                             "team_id": team, "opp": opp, "home": home, "goals": int(pid == 10 and i % 2 == 0), "assists": 1, "points": 1,
                             "sog": 3, "pp_goals": 0, "toi": 18.0})
    return pd.DataFrame(games), pd.DataFrame(rows)


def test_features_are_pre_game_and_future_rows_projected():
    games, sk = _world()
    panel = team_panel(games, pd.DataFrame())
    upcoming = games[games.status == "scheduled"]
    fut = pf.projected_lineups(sk, upcoming, panel)
    assert set(fut.player_id) == {10, 11, 20}
    feat = pf.build(sk, games, panel, fut)
    row = feat[(feat.player_id == 10) & (feat.game_id == 5)].iloc[0]
    # changing a later game must not move game-5 features; changing game 4 must
    sk2 = sk.copy()
    sk2.loc[(sk2.player_id == 10) & (sk2.game_id == 9), "goals"] = 5
    later = pf.build(sk2, games, panel, fut)
    assert np.isclose(later[(later.player_id == 10) & (later.game_id == 5)].iloc[0].goals_l5, row.goals_l5)
    sk3 = sk.copy()
    sk3.loc[(sk3.player_id == 10) & (sk3.game_id == 4), "goals"] = 5
    earlier = pf.build(sk3, games, panel, fut)
    assert earlier[(earlier.player_id == 10) & (earlier.game_id == 5)].iloc[0].goals_l5 > row.goals_l5
    # the projected row exists, has no stats, and has features
    nxt = feat[(feat.player_id == 10) & (feat.game_id == 29)].iloc[0]
    assert np.isnan(nxt.goals) and not np.isnan(nxt.goals_l10) and not np.isnan(nxt.opp_sa10)
    assert all(c in feat for c in pf.feature_columns())


def test_lineup_excludes_players_who_stopped_dressing():
    games, sk = _world()
    sk = sk[~((sk.player_id == 11) & (sk.game_id >= 26))]       # 11 missed the last 3 games
    panel = team_panel(games, pd.DataFrame())
    fut = pf.projected_lineups(sk, games[games.status == "scheduled"], panel)
    assert 11 not in set(fut.player_id) and 10 in set(fut.player_id)
