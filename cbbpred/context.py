"""Everything the JSON builders read, loaded once per export so each file is a pure function of it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

from cbbpred import config as cfg
from cbbpred.calls import prediction_frame
from cbbpred.espn import conferences
from cbbpred.form import TeamLogs
from cbbpred.lines import latest_lines, movement_index
from cbbpred.store import read_json, read_parquet

LOGO_URL = "https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"
PLAYED = ("final", "scheduled", "live")


@dataclass(frozen=True)
class Context:
    generated_at: datetime
    games: pl.DataFrame
    by_id: dict[str, dict[str, Any]]
    teams: dict[str, dict[str, Any]]
    confs: dict[str, dict[str, Any]]
    team_conf: dict[str, str]
    logs: TeamLogs
    calls: dict[str, dict[str, Any]]
    graded: dict[str, dict[str, Any]]
    lines: dict[str, dict[str, Any]]
    line_moves: dict[str, list[dict[str, Any]]]
    pre: dict[str, dict[str, Any]]
    team_ratings: dict[str, dict[str, Any]]
    power_rank: dict[str, int]
    ap: dict[str, Any]
    params: dict[str, Any]
    model_meta: dict[str, Any]

    def logo(self, team_id: str) -> str:
        return self.teams.get(team_id, {}).get("logo") or LOGO_URL.format(team_id=team_id)

    def is_d1(self, team_id: str) -> bool:
        return self.team_conf.get(team_id) in self.confs


def _by(df: pl.DataFrame, key: str) -> dict[str, dict[str, Any]]:
    return {r[key]: r for r in df.to_dicts()} if df.height else {}


def _json(path: Any) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _team_conf(games: pl.DataFrame) -> dict[str, str]:
    """Team -> conference: ESPN's list for this season, else the newest conference seen on a game."""
    seen: dict[str, str] = {}
    latest = games.sort("kickoff")
    for side in ("home", "away"):
        for tid, conf in latest.select(f"{side}_id", f"{side}_conf").drop_nulls().iter_rows():
            seen[tid] = conf
    return {**seen, **_json(cfg.team_conf_path())}


def load(now: datetime | None = None) -> Context:
    games_all = pl.read_parquet(cfg.games_path())
    games = games_all.filter(pl.col("status").is_in(PLAYED))
    finals = games.filter((pl.col("status") == "final") & pl.col("home_score").is_not_null())
    team_ratings = _by(read_parquet(cfg.team_ratings_path()), "team_id")
    team_conf = _team_conf(games)
    confs = conferences()
    d1_power = sorted((r["power"], tid) for tid, r in team_ratings.items() if team_conf.get(tid) in confs)
    power_rank = {tid: i + 1 for i, (_, tid) in enumerate(reversed(d1_power))}
    graded = read_parquet(cfg.graded_path())
    params_path = cfg.model_dir() / "rating_params.json"
    return Context(
        generated_at=now or datetime.now(UTC), games=games, by_id=_by(games, "game_id"),
        teams=_by(read_parquet(cfg.teams_path()), "team_id"), confs=confs, team_conf=team_conf,
        logs=TeamLogs(finals), calls=_by(prediction_frame(games), "game_id"), graded=_by(graded, "game_id"),
        lines=_by(latest_lines(), "game_id") if cfg.odds_history_path().exists() else {},
        line_moves=movement_index(),
        pre=_by(read_parquet(cfg.ratings_path()), "game_id"), team_ratings=team_ratings, power_rank=power_rank,
        ap=_json(cfg.ap_poll_path()), params=_json(params_path).get("params", {}),
        model_meta=_json(cfg.model_dir() / "metadata.json"),
    )


FLOAT_DIGITS = 4


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, FLOAT_DIGITS)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def compact(payload: Any) -> str:
    """Compact JSON with floats rounded; NaN or infinity raise (they would make the file unparseable in browsers)."""
    return json.dumps(_rounded(payload), separators=(",", ":"), default=str, allow_nan=False)
