"""Offense/defense points ratings, updated game by game.

Every team carries an offense rating (points scored above an average Division I team) and a defense
rating (points allowed above average, so lower is better). A game is projected as

    home points = base + hfa/2 + home offense + away defense
    away points = base - hfa/2 + away offense + home defense

After a final, each side's rating moves toward what the scoreboard showed (error capped, bigger steps
in a team's first games). Each game is projected *before* it updates anything, so the stored values
are what the model could have known at tip-off and the table is safe for training and backtesting.
Ratings are points per game; ``per100`` converts them with the average possessions per game.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any

import numpy as np
import polars as pl

from cbbpred import config as cfg

EARLY_GAMES = 6  # a team's first games of a season move its rating faster (rosters turn over every year)
PLAYED = ["final", "scheduled", "live"]


@dataclass(frozen=True)
class Params:
    k: float = 0.08
    k_early: float = 1.8
    hfa: float = 3.5
    carry: float = 0.6
    base: float = 72.0
    cap: float = 30.0
    unseen_off: float = -9.0  # a team outside Division I starts well below average
    unseen_def: float = 9.0


@dataclass
class Team:
    off: float
    deff: float
    season: int
    games: int = 0


def _prior(is_d1: bool, p: Params) -> tuple[float, float]:
    return (0.0, 0.0) if is_d1 else (p.unseen_off, p.unseen_def)


def _roll_season(teams: dict[str, Team], d1: dict[str, bool], season: int, p: Params) -> None:
    """New season: pull every rating back toward its class mean (D1 average, or the non-D1 prior)."""
    for tid, t in teams.items():
        o, d = _prior(d1[tid], p)
        t.off, t.deff = o + (t.off - o) * p.carry, d + (t.deff - d) * p.carry
        t.season, t.games = season, 0


def _team(teams: dict[str, Team], d1: dict[str, bool], tid: str, is_d1: bool, season: int, p: Params) -> Team:
    d1[tid] = d1.get(tid, False) or is_d1
    if tid not in teams:
        o, d = _prior(d1[tid], p)
        teams[tid] = Team(o, d, season)
    return teams[tid]


def project(home: Team, away: Team, neutral: bool, p: Params) -> tuple[float, float]:
    hfa = 0.0 if neutral else p.hfa
    return p.base + hfa / 2 + home.off + away.deff, p.base - hfa / 2 + away.off + home.deff


def update(home: Team, away: Team, ph: float, pa: float, hs: int, as_: int, p: Params) -> None:
    """Move both teams toward the final score. Mutates only the two private rating cells it is given."""
    err_h = float(np.clip(hs - ph, -p.cap, p.cap))
    err_a = float(np.clip(as_ - pa, -p.cap, p.cap))
    kh = p.k * (p.k_early if home.games < EARLY_GAMES else 1.0)
    ka = p.k * (p.k_early if away.games < EARLY_GAMES else 1.0)
    home.off, away.deff = home.off + kh * err_h, away.deff + kh * err_h
    away.off, home.deff = away.off + ka * err_a, home.deff + ka * err_a
    home.games += 1
    away.games += 1


def run(games: pl.DataFrame, d1_ids: set[str], p: Params = Params()) -> tuple[pl.DataFrame, dict[str, Team]]:
    """Walk every game in tip-off order. Returns pre-game projections and the final team states."""
    rows = games.filter(pl.col("status").is_in(PLAYED)).sort("kickoff", "game_id").to_dicts()
    teams: dict[str, Team] = {}
    d1: dict[str, bool] = {}
    out: list[dict[str, Any]] = []
    season: int | None = None
    for g in rows:
        if season is not None and g["season"] > season:
            _roll_season(teams, d1, g["season"], p)
        season = g["season"]
        home = _team(teams, d1, g["home_id"], g["home_conf"] in d1_ids, season, p)
        away = _team(teams, d1, g["away_id"], g["away_conf"] in d1_ids, season, p)
        ph, pa = project(home, away, bool(g["neutral"]), p)
        out.append({"game_id": g["game_id"], "home_off": home.off, "home_def": home.deff,
                    "away_off": away.off, "away_def": away.deff, "home_gp": home.games, "away_gp": away.games,
                    "proj_home_pts": ph, "proj_away_pts": pa, "rating_margin": ph - pa, "rating_total": ph + pa})
        if g["status"] == "final" and g["home_score"] is not None and g["away_score"] is not None:
            update(home, away, ph, pa, g["home_score"], g["away_score"], p)
    return pl.DataFrame(out, schema_overrides={"home_gp": pl.Int64, "away_gp": pl.Int64}), teams


def team_table(teams: dict[str, Team]) -> pl.DataFrame:
    """Current ratings: power = points better than an average Division I team on a neutral floor."""
    return pl.DataFrame([{"team_id": tid, "off": t.off, "def": t.deff, "power": t.off - t.deff,
                          "games": t.games, "season": t.season} for tid, t in teams.items()],
                        schema={"team_id": pl.Utf8, "off": pl.Float64, "def": pl.Float64, "power": pl.Float64,
                                "games": pl.Int64, "season": pl.Int64})


def per100(points_above_avg: float, base: float) -> float:
    """A rating in points per game above the base, as the team's points per 100 possessions."""
    return (base + points_above_avg) * 100.0 / cfg.AVG_POSSESSIONS


def margin_mae(games: pl.DataFrame, pre: pl.DataFrame, d1_ids: set[str], seasons: range | list[int]) -> float:
    ids = list(d1_ids)
    j = games.join(pre, on="game_id").filter(
        (pl.col("status") == "final") & pl.col("season").is_in(list(seasons))
        & pl.col("home_conf").is_in(ids) & pl.col("away_conf").is_in(ids))
    return float((j["home_score"] - j["away_score"] - j["rating_margin"]).abs().mean())


def tune(games: pl.DataFrame, d1_ids: set[str], fit_seasons: range, hold_seasons: range) -> dict[str, Any]:
    """Grid-search the update rule on ``fit_seasons`` and score the winner on seasons it never saw."""
    finals = games.filter((pl.col("status") == "final") & pl.col("season").is_in(list(fit_seasons)))
    base = replace(Params(), base=float(pl.concat([finals["home_score"], finals["away_score"]]).mean()))
    grid = product((0.05, 0.07, 0.09, 0.12), (1.0, 1.8, 2.6), (0.45, 0.6, 0.75), (2.8, 3.5, 4.2), (20.0, 30.0))
    scored = games.filter(pl.col("season") <= max(fit_seasons))
    best: tuple[float, Params] | None = None
    for k, early, carry, hfa, cap in grid:
        cand = replace(base, k=k, k_early=early, carry=carry, hfa=hfa, cap=cap)
        pre, _ = run(scored, d1_ids, cand)
        mae = margin_mae(scored, pre, d1_ids, range(min(fit_seasons) + 1, max(fit_seasons) + 1))
        if best is None or mae < best[0]:
            best = (mae, cand)
    if best is None:
        raise ValueError("empty tuning grid")
    pre, _ = run(games, d1_ids, best[1])
    return {"params": asdict(best[1]), "fit_mae": best[0], "fit_seasons": [min(fit_seasons), max(fit_seasons)],
            "holdout_mae": margin_mae(games, pre, d1_ids, hold_seasons),
            "holdout_seasons": [min(hold_seasons), max(hold_seasons)]}
