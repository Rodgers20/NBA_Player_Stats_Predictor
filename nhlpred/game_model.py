"""Game-winner model: Elo + rolling form + rest + goalie + home ice -> logistic regression, plus goal projections.

Walk-forward: for every calendar month in the test span the model is refit on games that finished BEFORE that month.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nhlpred import config as cfg
from nhlpred import evaluate as ev
from nhlpred.elo import EloParams, run_elo
from nhlpred.elo import tune as tune_elo
from nhlpred.store import read_json, read_parquet, write_json
from nhlpred.teamfeat import team_panel

log = logging.getLogger(__name__)

FEATURES = ["f_elo", "f_gd10", "f_gd30", "f_sd10", "f_sd30", "f_win10", "f_goalie", "rest_h", "rest_a", "b2b_h", "b2b_a"]
LABELS = {"f_elo": "Team strength (Elo)", "f_gd10": "Goal differential, last 10", "f_gd30": "Goal differential, last 30",
          "f_sd10": "Shot differential, last 10", "f_sd30": "Shot differential, last 30", "f_win10": "Win rate, last 10",
          "f_goalie": "Starting goalie save %", "rest_h": "Home rest", "rest_a": "Away rest", "b2b_h": "Home back-to-back",
          "b2b_a": "Away back-to-back"}
GOAL_FEATS_H = ["gf30_h", "ga30_a", "sf30_h", "sa30_a", "gf10_h", "ga10_a", "svp_a", "b2b_h", "b2b_a"]
GOAL_FEATS_A = ["gf30_a", "ga30_h", "sf30_a", "sa30_h", "gf10_a", "ga10_h", "svp_h", "b2b_a", "b2b_h"]
TEST_FROM_SEASON = 2025   # 2023-24 is warm-up and training only


def load_games() -> pd.DataFrame:
    g = read_parquet(cfg.games_path())
    return g.sort_values(["start_utc", "game_id"]).reset_index(drop=True)


def build_frame(games: pd.DataFrame, goalies: pd.DataFrame, params: EloParams | None = None) -> pd.DataFrame:
    """One feature row per game, finished or not; every value is pre-game."""
    params = params or EloParams()
    elo, _ = run_elo(games, params)
    panel = team_panel(games, goalies)
    keep = ["game_id", "team_id", "rest", "b2b", "gf10", "ga10", "sf10", "sa10", "win10", "gf30", "ga30", "sf30", "sa30", "svp",
            "goalie_id"] + (["goalie_projected"] if "goalie_projected" in panel else [])
    cols = [c for c in keep if c in panel]
    h = panel[cols].add_suffix("_h").rename(columns={"game_id_h": "game_id", "team_id_h": "home_id"})
    a = panel[cols].add_suffix("_a").rename(columns={"game_id_a": "game_id", "team_id_a": "away_id"})
    f = games.merge(elo, on="game_id").merge(h, on=["game_id", "home_id"]).merge(a, on=["game_id", "away_id"])
    f["y"] = np.where(f.home_score.notna(), (f.home_score > f.away_score).astype(float), np.nan)
    f["f_elo"] = f.elo_h - f.elo_a
    f["f_gd10"] = (f.gf10_h - f.ga10_h) - (f.gf10_a - f.ga10_a)
    f["f_gd30"] = (f.gf30_h - f.ga30_h) - (f.gf30_a - f.ga30_a)
    f["f_sd10"] = (f.sf10_h - f.sa10_h) - (f.sf10_a - f.sa10_a)
    f["f_sd30"] = (f.sf30_h - f.sa30_h) - (f.sf30_a - f.sa30_a)
    f["f_win10"] = f.win10_h - f.win10_a
    f["f_goalie"] = (f.svp_h - f.svp_a) * 100
    f["rest_h"], f["rest_a"] = f.rest_h.clip(upper=4), f.rest_a.clip(upper=4)
    return f.sort_values(["start_utc", "game_id"]).reset_index(drop=True)


def _lr() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0))


def fit(train: pd.DataFrame) -> dict[str, Any]:
    done = train[train.y.notna()]
    return {"win": _lr().fit(done[FEATURES], done.y),
            "elo_only": LogisticRegression().fit(done[["f_elo"]], done.y),
            "home_rate": float(done.y.mean()),
            "goals_h": _ridge().fit(done[GOAL_FEATS_H], done.home_score),
            "goals_a": _ridge().fit(done[GOAL_FEATS_A], done.away_score),
            "mean_total": float((done.home_score + done.away_score).mean())}


def predict(models: dict[str, Any], f: pd.DataFrame) -> pd.DataFrame:
    out = f[["game_id"]].copy()
    out["p_home"] = models["win"].predict_proba(f[FEATURES])[:, 1]
    out["p_elo"] = models["elo_only"].predict_proba(f[["f_elo"]])[:, 1]
    out["proj_home"] = np.clip(models["goals_h"].predict(f[GOAL_FEATS_H]), 1.0, 6.0)
    out["proj_away"] = np.clip(models["goals_a"].predict(f[GOAL_FEATS_A]), 1.0, 6.0)
    out["proj_total"] = out.proj_home + out.proj_away
    return out


def walk_forward(frame: pd.DataFrame, test_from: int = TEST_FROM_SEASON) -> pd.DataFrame:
    """Out-of-sample predictions for every finished game from ``test_from`` on, refit monthly."""
    done = frame[frame.y.notna()].copy()
    done["month"] = pd.to_datetime(done.date).dt.to_period("M")
    outs = []
    for month in sorted(done[done.season >= test_from].month.unique()):
        train, test = done[done.month < month], done[done.month == month]
        if len(train) < 500:
            continue
        p = predict(fit(train), test)
        p["home_rate"] = float(train.y.mean())
        p["mean_total"] = float((train.home_score + train.away_score).mean())
        outs.append(p)
    return pd.concat(outs, ignore_index=True)


def backtest_summary(frame: pd.DataFrame, preds: pd.DataFrame) -> dict[str, Any]:
    m = preds.merge(frame[["game_id", "season", "y", "home_score", "away_score"]], on="game_id")
    total = m.home_score + m.away_score

    def block(d: pd.DataFrame) -> dict[str, Any]:
        t = d.home_score + d.away_score
        base_ll = ev.log_loss(d.y, d.home_rate)
        return {
            "n": int(len(d)),
            "model": {"accuracy": round(ev.accuracy(d.y, d.p_home), 4), **ev.summary(d.y, d.p_home)},
            "baselines": {
                "home_always": {"accuracy": round(float(d.y.mean()), 4), **ev.summary(d.y, d.home_rate)},
                "elo_only": {"accuracy": round(ev.accuracy(d.y, d.p_elo), 4), **ev.summary(d.y, d.p_elo)}},
            "total_goals": {"mae": round(float((d.proj_total - t).abs().mean()), 3),
                            "mae_baseline": round(float((d.mean_total - t).abs().mean()), 3),
                            "mean_actual": round(float(t.mean()), 3), "mean_pred": round(float(d.proj_total.mean()), 3)},
            "home_win_rate": round(float(d.y.mean()), 4), "_ll_base": base_ll}

    overall = block(m)
    overall.pop("_ll_base")
    overall["calibration"] = ev.calibration(m.y, m.p_home)
    overall["by_season"] = {cfg.season_label(int(s)): {k: v for k, v in block(d).items() if k in ("n", "model", "baselines")}
                            for s, d in m.groupby("season")}
    overall["seasons"] = [cfg.season_label(int(s)) for s in sorted(m.season.unique())]
    overall["pct_high_conf"] = round(float(((m.p_home > 0.62) | (m.p_home < 0.38)).mean()), 4)
    hc = m[(m.p_home > 0.62) | (m.p_home < 0.38)]
    overall["high_conf_accuracy"] = round(ev.accuracy(hc.y, hc.p_home), 4) if len(hc) else None
    overall["note"] = ("NHL games are close to coin flips: even the market rarely beats ~60% straight-up. "
                       "Treat the model as a small edge over home-ice-only, not a lock.")
    return overall


def version(now: datetime | None = None) -> str:
    return "nhl-game-" + (now or datetime.now(UTC)).strftime("%Y%m%d")


def train(now: datetime | None = None, backtest: bool = True) -> dict[str, Any]:
    """Tune Elo on the first season, run the walk-forward backtest, fit the production model on everything."""
    games = load_games()
    goalies = read_parquet(cfg.goalies_path())
    params = tune_elo(games, cfg.FIRST_SEASON)
    log.info("elo params %s", params)
    frame = build_frame(games, goalies, params)
    summary: dict[str, Any] = {}
    if backtest:
        preds = walk_forward(frame)
        summary = backtest_summary(frame, preds)
        preds.merge(frame[["game_id", "date", "home_abbr", "away_abbr", "y"]], on="game_id").to_parquet(cfg.data_dir() / "game_backtest.parquet", index=False)
    models = fit(frame)
    now = now or datetime.now(UTC)
    bundle = {"models": models, "elo": params.__dict__, "version": version(now), "trained_at": now.isoformat(),
              "train_games": int(frame.y.notna().sum())}
    cfg.model_dir().mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cfg.model_dir() / "game_model.joblib")
    write_json({"version": bundle["version"], "trained_at": bundle["trained_at"], "train_games": bundle["train_games"],
                "elo": bundle["elo"], "backtest": summary}, cfg.model_dir() / "game_summary.json", compact=False)
    return bundle


def load_bundle() -> dict[str, Any]:
    return joblib.load(cfg.model_dir() / "game_model.joblib")


def summary_json() -> dict[str, Any]:
    return read_json(cfg.model_dir() / "game_summary.json", {})


def explain(models: dict[str, Any], row: pd.Series) -> list[dict[str, Any]]:
    """Per-factor push on home win probability (percentage points) vs. an average game, from the logistic coefficients."""
    pipe = models["win"]
    scaler, lr = pipe.steps[0][1], pipe.steps[1][1]
    z = (row[FEATURES].astype(float).values - scaler.mean_) / scaler.scale_
    contrib = lr.coef_[0] * z
    p = float(pipe.predict_proba(row[FEATURES].astype(float).to_frame().T)[:, 1][0])
    return [{"key": k, "pct": float(p * (1 - p) * c * 100)} for k, c in zip(FEATURES, contrib)]
