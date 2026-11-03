"""Everything the exporters need, computed once: features, predictions, team and player lookups."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nhlpred import config as cfg
from nhlpred import game_model as gm
from nhlpred import pipeline
from nhlpred import predict as pr
from nhlpred import prop_model as pm
from nhlpred.elo import EloParams, run_elo
from nhlpred.store import read_json, read_parquet

EASTERN = ZoneInfo("America/New_York")


@dataclass
class Context:
    now: datetime
    today: date
    data: pipeline.Data
    teams: dict[int, dict[str, Any]]
    frame: pd.DataFrame                  # one pre-game feature row per game, indexed by game_id
    gpred: pd.DataFrame                  # p_home / proj_* / source per game, indexed by game_id
    gbundle: dict[str, Any]
    pbundle: dict[str, Any]
    elo_now: dict[int, float]
    proj: pd.DataFrame                   # projected-lineup rows with features (upcoming games)
    pprob: dict[str, dict[str, Any]]     # prop predictions aligned to ``proj``
    players: pd.DataFrame                # finished skater-games, newest last
    standings: list[dict[str, Any]]
    goalie_names: dict[int, str] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def stamp(self) -> str:
        return self.now.strftime("%Y-%m-%dT%H:%MZ")


def _game_predictions(frame: pd.DataFrame, games: pd.DataFrame, bundle: dict[str, Any], now: datetime) -> pd.DataFrame:
    """Live model for unplayed games; frozen log for started ones; walk-forward replay for older finals."""
    live = gm.predict(bundle["models"], frame)
    live["source"] = "live"
    bt = read_parquet(cfg.data_dir() / "game_backtest.parquet")
    parts = []
    if not bt.empty:
        b = bt[["game_id", "p_home", "proj_home", "proj_away", "proj_total"]].copy()
        b["source"] = "backtest"
        parts.append(b)
    parts.append(live[live.game_id.isin(games[games.status != "final"].game_id)])
    out = pd.concat(parts, ignore_index=True).drop_duplicates("game_id", keep="last").set_index("game_id")
    frozen = read_parquet(cfg.data_dir() / pr.GAME_LOG)
    if not frozen.empty:                      # a started game keeps the call it had at puck drop
        f = frozen[frozen.game_id.isin(pr.started_games(games, now))].set_index("game_id")[["p_home", "proj_home", "proj_away"]]
        f = f.assign(proj_total=f.proj_home + f.proj_away, source="live")
        out = f.combine_first(out)
    return out


def load(now: datetime | None = None) -> Context:
    now = now or datetime.now(UTC)
    today = now.astimezone(EASTERN).date()
    data = pipeline.load()
    gbundle, pbundle = gm.load_bundle(), pm.load_bundle()
    params = EloParams(**gbundle["elo"])
    frame = gm.build_frame(data.games, data.goalies, params)
    _, elo_now = run_elo(data.games, params)
    feat = pipeline.player_table(data, today)
    proj = feat[feat.projected & feat.game_id.isin(pipeline.upcoming_games(data.games, today).game_id)].reset_index(drop=True)
    pprob = pm.predict(pbundle, proj) if len(proj) else {}
    teams = {t["id"]: t for t in read_json(cfg.teams_path(), [])}
    st = read_json(cfg.standings_path(), {"standings": []})["standings"]
    names = {}
    if not data.goalies.empty:
        names = data.goalies.sort_values("date").drop_duplicates("player_id", keep="last").set_index("player_id")["name"].to_dict()
    return Context(now, today, data, teams, frame.set_index("game_id", drop=False), _game_predictions(frame, data.games, gbundle, now),
                   gbundle, pbundle, elo_now, proj, pprob, feat[~feat.projected], st, names)


def team_ref(ctx: Context, team_id: int) -> dict[str, Any]:
    t = ctx.teams[int(team_id)]
    return {k: t[k] for k in ("id", "abbr", "name", "short", "logo", "logo_dark")}


def confidence(p_home: float) -> str:
    p = max(p_home, 1 - p_home)
    return "high" if p > 0.62 else "medium" if p >= 0.56 else "low"


def nan_to_none(v: Any) -> Any:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
