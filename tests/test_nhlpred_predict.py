from datetime import UTC, datetime

import pandas as pd

from nhlpred import config as cfg
from nhlpred import predict as pr


def _games():
    return pd.DataFrame({"game_id": [1, 2], "status": ["scheduled", "scheduled"], "date": ["d", "d"],
                         "start_utc": ["2026-11-03T00:00:00Z", "2026-11-05T00:00:00Z"], "home_abbr": ["A", "C"], "away_abbr": ["B", "D"],
                         "home_score": [None, None], "away_score": [None, None]})


def test_prediction_freezes_at_puck_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_dir", lambda: tmp_path)
    g = _games()
    p1 = pd.DataFrame({"game_id": [1, 2], "p_home": [0.6, 0.55], "proj_home": [3, 3], "proj_away": [2, 2]})
    pr.log_game_predictions(g, p1, "v1", datetime(2026, 11, 2, tzinfo=UTC))
    p2 = p1.assign(p_home=[0.9, 0.9])
    pr.log_game_predictions(g, p2, "v2", datetime(2026, 11, 3, 1, tzinfo=UTC))      # game 1 started, game 2 has not
    log = pd.read_parquet(tmp_path / pr.GAME_LOG).set_index("game_id")
    assert log.loc[1, "p_home"] == 0.6 and log.loc[1, "model_version"] == "v1"     # frozen
    assert log.loc[2, "p_home"] == 0.9 and log.loc[2, "model_version"] == "v2"     # still live


def test_grade_games(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "data_dir", lambda: tmp_path)
    g = _games()
    pr.log_game_predictions(g, pd.DataFrame({"game_id": [1, 2], "p_home": [0.6, 0.4], "proj_home": [3, 3], "proj_away": [2, 2]}), "v", datetime(2026, 11, 2, tzinfo=UTC))
    g.loc[:, "status"] = "final"
    g["home_score"], g["away_score"] = [3, 3], [1, 2]
    out = pr.grade_games(g)
    assert out.correct.tolist() == [True, False]
