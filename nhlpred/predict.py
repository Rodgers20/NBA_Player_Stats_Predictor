"""Frozen pre-game call logs and grading.

A game's prediction keeps updating until puck drop, then freezes, so a late projection can never be rewritten with
hindsight. Props log only the main line of each market (the full line ladder is derived at export time).
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nhlpred import config as cfg
from nhlpred import prop_model as pm
from nhlpred.store import read_parquet, write_parquet

GAME_LOG = "game_predictions.parquet"
PROP_LOG = "prop_predictions.parquet"


def _upsert(path, new: pd.DataFrame, keys: list[str], started: set) -> int:
    """Log calls for games that have not started; rows already logged for a started game are never touched."""
    new = new[~new.game_id.isin(started)]
    old = read_parquet(path)
    if not old.empty:
        old = old[~old.game_id.isin(set(new.game_id))]
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new
    write_parquet(out.drop_duplicates(keys, keep="last"), path)
    return len(new)


def started_games(games: pd.DataFrame, now: datetime) -> set:
    t = pd.to_datetime(games.start_utc, utc=True)
    return set(games[t <= pd.Timestamp(now)].game_id)


def log_game_predictions(games: pd.DataFrame, preds: pd.DataFrame, version: str, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    up = preds[preds.game_id.isin(games[games.status != "final"].game_id)].copy()
    up["made_at"], up["model_version"] = now.strftime("%Y-%m-%dT%H:%MZ"), version
    keep = up[["game_id", "made_at", "model_version", "p_home", "proj_home", "proj_away"]]
    return _upsert(cfg.data_dir() / GAME_LOG, keep, ["game_id"], started_games(games, now))


def log_prop_predictions(rows: pd.DataFrame, pred: dict, games: pd.DataFrame, version: str, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    parts = []
    for market, m in pm.MARKETS.items():
        parts.append(pd.DataFrame({"game_id": rows.game_id.values, "player_id": rows.player_id.values, "market": market, "line": m["main"],
                                   "p": pred[market]["p"][m["main"]], "mu": pred[market]["mu"]}))
    new = pd.concat(parts, ignore_index=True)
    new["made_at"], new["model_version"] = now.strftime("%Y-%m-%dT%H:%MZ"), version
    return _upsert(cfg.data_dir() / PROP_LOG, new, ["game_id", "player_id", "market"], started_games(games, now))


def grade_games(games: pd.DataFrame) -> pd.DataFrame:
    log = read_parquet(cfg.data_dir() / GAME_LOG)
    if log.empty:
        return log
    fin = games[games.status == "final"][["game_id", "date", "home_abbr", "away_abbr", "home_score", "away_score"]]
    g = fin.merge(log, on="game_id")
    g["home_won"] = g.home_score > g.away_score
    g["correct"] = (g.p_home >= 0.5) == g.home_won
    return g.sort_values("date").reset_index(drop=True)


def grade_props(skaters: pd.DataFrame) -> pd.DataFrame:
    log = read_parquet(cfg.data_dir() / PROP_LOG)
    if log.empty:
        return log
    actual = skaters[["game_id", "player_id", "date", "name", "team", "goals", "assists", "points", "sog"]]
    g = log.merge(actual, on=["game_id", "player_id"])
    target = {m: v["target"] for m, v in pm.MARKETS.items()}
    g["count"] = [row[target[row["market"]]] for row in g[["market", "goals", "assists", "points", "sog"]].to_dict("records")]
    g["hit"] = np.where(g["count"] > g.line, 1.0, 0.0)
    return g


def props_record(graded: pd.DataFrame) -> dict:
    out = {}
    for market in pm.MARKETS:
        g = graded[graded.market == market] if not graded.empty else graded
        if g is None or g.empty:
            out[market] = {"graded": 0, "hit_rate": None, "avg_p": None, "top_decile_hit_rate": None}
            continue
        k = max(1, len(g) // 10)
        top = g.nlargest(k, "p")
        out[market] = {"graded": int(len(g)), "hit_rate": round(float(g.hit.mean()), 4), "avg_p": round(float(g.p.mean()), 4),
                       "top_decile_hit_rate": round(float(top.hit.mean()), 4)}
    return out
