import numpy as np
import pandas as pd

from nhlpred import game_model as gm


def _frame(n=1800, seed=1):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({c: rng.normal(size=n) for c in gm.FEATURES})
    f["f_elo"] = rng.normal(0, 60, n)
    p = 1 / (1 + np.exp(-(0.25 + f.f_elo / 150)))
    f["y"] = (rng.random(n) < p).astype(float)
    f["home_score"] = np.where(f.y == 1, 4, 2) + rng.integers(0, 2, n)
    f["away_score"] = 6 - f.home_score + 1
    for c in gm.GOAL_FEATS_H + gm.GOAL_FEATS_A:
        f[c] = rng.normal(3, 0.3, n) if c not in f else f[c]
    f["game_id"] = np.arange(n)
    f["season"] = np.where(np.arange(n) < 900, 2025, 2026)
    start = pd.Timestamp("2025-10-01") + pd.to_timedelta(np.arange(n) // 6, unit="D")
    f["date"] = start.strftime("%Y-%m-%d")
    return f


def test_walk_forward_beats_home_always_when_there_is_signal_and_never_trains_on_the_future():
    f = _frame()
    preds = gm.walk_forward(f, test_from=2025)
    assert preds.game_id.is_unique and len(preds) > 800
    s = gm.backtest_summary(f, preds)
    assert s["model"]["log_loss"] < s["baselines"]["home_always"]["log_loss"]
    assert s["n"] == len(preds) and 0 < s["calibration"][0]["predicted"] < s["calibration"][-1]["predicted"] < 1
    # a game's prediction must not move when its OWN result (or later results) change
    g = f.copy()
    late = g.date >= g.date.iloc[1200]
    g.loc[late, "y"] = 1 - g.loc[late, "y"]
    g.loc[late, ["home_score", "away_score"]] = g.loc[late, ["away_score", "home_score"]].values
    p2 = gm.walk_forward(g, test_from=2025)
    cut = f.date.iloc[1200]
    month = pd.to_datetime(cut).to_period("M")
    early_ids = f[pd.to_datetime(f.date).dt.to_period("M") < month].game_id
    early_ids = early_ids[early_ids.isin(preds.game_id)]
    assert len(early_ids) > 100
    a, b = preds.set_index("game_id").loc[early_ids], p2.set_index("game_id").loc[early_ids]
    assert np.allclose(a.p_home, b.p_home)


def test_explain_pushes_toward_the_stronger_home_side():
    f = _frame()
    models = gm.fit(f)
    row = f.iloc[0].copy()
    row["f_elo"] = 150.0
    elo_push = {c["key"]: c["pct"] for c in gm.explain(models, row)}["f_elo"]
    assert elo_push > 0
