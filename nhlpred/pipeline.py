"""Glue: load the tables and build the feature frames every later step shares."""

from __future__ import annotations

from datetime import date, timedelta
from typing import NamedTuple

import pandas as pd

from nhlpred import config as cfg
from nhlpred import game_model as gm
from nhlpred import playerfeat as pf
from nhlpred.store import read_parquet
from nhlpred.teamfeat import team_panel


class Data(NamedTuple):
    games: pd.DataFrame
    skaters: pd.DataFrame
    goalies: pd.DataFrame
    panel: pd.DataFrame


def load() -> Data:
    games = gm.load_games()
    sk, gl = read_parquet(cfg.skaters_path()), read_parquet(cfg.goalies_path())
    return Data(games, sk, gl, team_panel(games, gl))


def upcoming_games(games: pd.DataFrame, today: date, days: int = cfg.EXPORT_DAYS_AHEAD) -> pd.DataFrame:
    end = (today + timedelta(days=days)).isoformat()
    return games[(games.status != "final") & (games.date >= today.isoformat()) & (games.date <= end)]


def player_table(d: Data, today: date | None = None, with_future: bool = True) -> pd.DataFrame:
    """Feature rows for every played skater-game, plus projected-lineup rows for upcoming games."""
    fut = None
    if with_future:
        up = upcoming_games(d.games, today or date.today())
        fut = pf.projected_lineups(d.skaters, up, d.panel) if len(up) else None
    feat = pf.build(d.skaters, d.games, d.panel, fut)
    feat["projected"] = feat.goals.isna()
    return feat


def train_all(now=None) -> dict:
    """Retrain both model families (with their walk-forward backtests) on everything ingested so far."""
    from nhlpred import prop_model as pm
    game = gm.train(now)
    df = player_table(load(), with_future=False)
    res = pm.walk_forward(df)
    props = pm.train(df, pm.backtest_summary(res), now)
    return {"game": game["version"], "props": props["version"]}


def models_stale(now=None) -> bool:
    from datetime import UTC, datetime

    from nhlpred import prop_model as pm
    now = now or datetime.now(UTC)
    try:
        trained = [datetime.fromisoformat(s["trained_at"]) for s in (gm.summary_json(), pm.summary_json())]
    except KeyError:
        return True
    return (now - min(trained)).days >= cfg.RETRAIN_DAYS
