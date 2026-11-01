"""Stage-2 model: rating outputs + schedule context -> projected margin and total.

``ratings.run`` gives a pre-game projection for every game. A small ridge correction on top (rest,
games played, offense/defense split, home court) is measured with a walk-forward backtest: each
season is predicted by a model that only saw earlier seasons. Training and scoring use only games
between two Division I teams; games against smaller schools still get a projection for display.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cbbpred import config as cfg
from cbbpred import ratings
from cbbpred.espn import conferences
from cbbpred.evaluate import calibration, scores, vs_market
from cbbpred.lines import latest_lines
from cbbpred.store import read_json, write_json, write_parquet

log = logging.getLogger(__name__)

ALPHA = 30.0
HALF_LIFE_DAYS = 365.0   # older games count less: scoring and home edge drift
REST_CAP_DAYS = 10.0
REST_DEFAULT_DAYS = 7.0  # a team with no earlier game on record
FEATURES = ["rating_margin", "rating_total", "home_off", "home_def", "away_off", "away_def", "neutral",
            "conf_game", "home_gp", "away_gp", "home_rest", "away_rest", "day_of_season"]
LINE_COLUMNS = ["spread_line", "total_line", "home_moneyline", "away_moneyline"]


def load_params() -> ratings.Params:
    path = cfg.model_dir() / "rating_params.json"
    return ratings.Params(**read_json(path)["params"]) if path.exists() else ratings.Params()


def d1_ids() -> set[str]:
    ids = set(conferences())
    if not ids:
        raise FileNotFoundError(f"No conference list at {cfg.conferences_path()}. Run `python -m cbbpred ingest`.")
    return ids


def rest_days(games: pl.DataFrame) -> pl.DataFrame:
    """Days since each side's previous game (capped, a default when it has none on record)."""
    live = games.filter(pl.col("status").is_in(ratings.PLAYED))
    sides = pl.concat([
        live.select("game_id", "kickoff", team=pl.col("home_id"), side=pl.lit("home")),
        live.select("game_id", "kickoff", team=pl.col("away_id"), side=pl.lit("away")),
    ]).sort("team", "kickoff")
    gap = (pl.col("kickoff") - pl.col("kickoff").shift(1).over("team")).dt.total_hours() / 24
    sides = sides.with_columns(rest=gap.clip(upper_bound=REST_CAP_DAYS).fill_null(REST_DEFAULT_DAYS))
    home = sides.filter(pl.col("side") == "home").select("game_id", home_rest="rest")
    away = sides.filter(pl.col("side") == "away").select("game_id", away_rest="rest")
    return home.join(away, on="game_id")


def feature_table(games: pl.DataFrame, pre: pl.DataFrame, ids: set[str]) -> pl.DataFrame:
    """One row per game: context + pre-game ratings + rest. Targets are null until the game is final."""
    keep = games.select("game_id", "season", "kickoff", "gameday", "status", "neutral", "conf_game",
                        "home_score", "away_score", "home_conf", "away_conf")
    season_open = pl.date(pl.col("season") - 1, *cfg.SEASON_OPEN)
    both_d1 = pl.col("home_conf").is_in(list(ids)) & pl.col("away_conf").is_in(list(ids))
    table = keep.join(pre, on="game_id").join(rest_days(games), on="game_id")
    return table.with_columns(
        pl.col("neutral").cast(pl.Float64), pl.col("conf_game").cast(pl.Float64),
        day_of_season=(pl.col("kickoff").dt.date() - season_open).dt.total_days().cast(pl.Float64),
        both_d1=both_d1, margin=pl.col("home_score") - pl.col("away_score"),
        total=pl.col("home_score") + pl.col("away_score"),
    ).drop("home_conf", "away_conf")


def refresh_ratings() -> pl.DataFrame:
    """Rebuild the pre-game ratings table and the current team ratings from every game on disk."""
    games = pl.read_parquet(cfg.games_path())
    ids = d1_ids()
    pre, teams = ratings.run(games, ids, load_params())
    table = feature_table(games, pre, ids)
    write_parquet(table, cfg.ratings_path())
    write_parquet(ratings.team_table(teams), cfg.team_ratings_path())
    return table


def recency_weights(train: pl.DataFrame, asof: datetime) -> np.ndarray:
    age = (asof - train["kickoff"]).dt.total_seconds().to_numpy() / 86400
    return 0.5 ** (age / HALF_LIFE_DAYS)


def _fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(x, y, ridge__sample_weight=weights)


def matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).to_numpy().astype(float)


def _finals(table: pl.DataFrame) -> pl.DataFrame:
    return table.filter((pl.col("status") == "final") & pl.col("both_d1")).drop_nulls(["margin", "total"])


def walk_forward(table: pl.DataFrame) -> pl.DataFrame:
    """Predict each season from models fit on earlier seasons only (the first season is burn-in)."""
    finals = _finals(table)
    out = []
    for season in range(cfg.FIRST_BACKTEST_SEASON, int(finals["season"].max()) + 1):
        train = finals.filter((pl.col("season") > cfg.FIRST_SEASON) & (pl.col("season") < season))
        test = finals.filter(pl.col("season") == season)
        if test.is_empty() or train.is_empty():
            continue
        w = recency_weights(train, test["kickoff"].min())
        mm = _fit(matrix(train), train["margin"].to_numpy(), w)
        mt = _fit(matrix(train), train["total"].to_numpy(), w)
        out.append(test.select("game_id", "season", "rating_margin", "rating_total", "margin", "total")
                   .with_columns(pred_margin=pl.Series(mm.predict(matrix(test))),
                                 pred_total=pl.Series(mt.predict(matrix(test)))))
    if not out:
        raise ValueError("no season has earlier seasons to train on; ingest more history")
    return pl.concat(out)


def _block(bt: pl.DataFrame, sigma: float) -> dict[str, Any]:
    from scipy.stats import norm
    p = norm.cdf(bt["pred_margin"].to_numpy() / sigma)
    return {"n_games": bt.height, "seasons": [int(bt["season"].min()), int(bt["season"].max())],
            "model": scores(bt, "pred_margin", "pred_total", sigma),
            "ratings_only": scores(bt, "rating_margin", "rating_total", sigma),
            "home_team_always_wins": float((bt["margin"] > 0).mean()),
            "calibration": calibration(p, (bt["margin"].to_numpy() > 0).astype(float))}


def summarize(bt: pl.DataFrame, holdout_from: int | None = None) -> dict[str, Any]:
    """Backtest scores. The rating update rule was tuned on seasons before ``holdout_from``, so the
    ``holdout`` block is the honest headline; the full block is optimistic."""
    sigma = float((bt["margin"] - bt["pred_margin"]).std())
    sigma_total = float((bt["total"] - bt["pred_total"]).std())
    by_season = [{"season": int(s), "n": r.height, **scores(r, "pred_margin", "pred_total", sigma)}
                 for (s,), r in sorted(bt.group_by("season"), key=lambda g: g[0][0])]
    held = bt.filter(pl.col("season") >= holdout_from) if holdout_from is not None else bt.clear()
    with_lines = bt.join(latest_lines().select("game_id", *LINE_COLUMNS), on="game_id", how="left") \
        if cfg.odds_history_path().exists() else bt
    return {**_block(bt, sigma), "sigma_margin": sigma, "sigma_total": sigma_total, "by_season": by_season,
            "holdout": {**_block(held, sigma), "from_season": holdout_from} if held.height else None,
            "vs_market": vs_market(with_lines)}


def backtest() -> dict[str, Any]:
    table = pl.read_parquet(cfg.ratings_path())
    bt = walk_forward(table)
    write_parquet(bt, cfg.backtest_path())
    tuned = cfg.model_dir() / "rating_params.json"
    holdout_from = read_json(tuned)["holdout_seasons"][0] if tuned.exists() else None
    summary = summarize(bt, holdout_from)
    write_json(summary, cfg.model_dir() / "backtest_summary.json")
    return summary


def train() -> dict[str, Any]:
    """Fit the production model on every Division I final since the burn-in season."""
    table = pl.read_parquet(cfg.ratings_path())
    finals = _finals(table).filter(pl.col("season") > cfg.FIRST_SEASON)
    summary = read_json(cfg.model_dir() / "backtest_summary.json")
    version = f"cbb-{datetime.now(UTC):%Y%m%d}"
    w = recency_weights(finals, datetime.now(UTC))
    artifact = {
        "version": version, "features": FEATURES,
        "margin": _fit(matrix(finals), finals["margin"].to_numpy(), w),
        "total": _fit(matrix(finals), finals["total"].to_numpy(), w),
        "sigma_margin": summary["sigma_margin"], "sigma_total": summary["sigma_total"],
    }
    cfg.model_dir().mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, cfg.model_dir() / "model.joblib")
    meta = {"version": version, "trained_at": datetime.now(UTC).isoformat(), "n_games": finals.height,
            "seasons": [int(finals["season"].min()), int(finals["season"].max())],
            "rating_params": asdict(load_params())}
    write_json(meta, cfg.model_dir() / "metadata.json")
    return meta


def load_model() -> dict[str, Any]:
    path = cfg.model_dir() / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No trained CBB model at {path}. Run `python -m cbbpred train`.")
    return joblib.load(path)  # trusted: written only by train() into our own model dir, never downloaded
