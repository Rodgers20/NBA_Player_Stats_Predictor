"""Elo ratings, walk-forward: the rating a game sees is computed only from games before it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = 1500.0
DEFAULTS = {"k": 6.0, "hfa": 35.0, "regress": 0.30}   # k/hfa chosen on season 2024 only, see tune()


@dataclass(frozen=True)
class EloParams:
    k: float = DEFAULTS["k"]
    hfa: float = DEFAULTS["hfa"]
    regress: float = DEFAULTS["regress"]


def expected_home(diff: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(diff + hfa) / 400.0))


def run_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> tuple[pd.DataFrame, dict[int, float]]:
    """Pre-game ratings for every game in ``games`` (sorted by date), plus final ratings.

    Unplayed games (no score) get the ratings as of the last played game, and do not update anything.
    """
    r: dict[int, float] = {}
    season = None
    pre_h, pre_a = np.empty(len(games)), np.empty(len(games))
    for i, g in enumerate(games.itertuples(index=False)):
        if season is not None and g.season != season:
            r = {t: BASE + (v - BASE) * (1 - params.regress) for t, v in r.items()}
        season = g.season
        rh, ra = r.get(g.home_id, BASE), r.get(g.away_id, BASE)
        pre_h[i], pre_a[i] = rh, ra
        if pd.isna(g.home_score):
            continue
        diff = rh - ra
        won = g.home_score > g.away_score
        margin = abs(g.home_score - g.away_score)
        winner_diff = (diff + params.hfa) if won else -(diff + params.hfa)
        mult = math.log(margin + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = params.k * mult * ((1.0 if won else 0.0) - expected_home(diff, params.hfa))
        r[g.home_id], r[g.away_id] = rh + delta, ra - delta
    out = games[["game_id"]].copy()
    out["elo_h"], out["elo_a"] = pre_h, pre_a
    return out, r


def tune(games: pd.DataFrame, season: int, grid_k=(3, 5, 8, 12), grid_hfa=(20, 35, 50), warmup: int = 300) -> EloParams:
    """Pick k / hfa by log loss on ONE season (the first), so later seasons stay a clean test."""
    games = games[games.season <= season]
    best, best_ll = EloParams(), 9.0
    for k in grid_k:
        for hfa in grid_hfa:
            p = EloParams(k, hfa)
            e, _ = run_elo(games, p)
            m = games.assign(elo_h=e.elo_h.values, elo_a=e.elo_a.values)
            m = m[(m.season == season) & m.home_score.notna()].iloc[warmup:]
            ph = 1 / (1 + 10 ** (-(m.elo_h - m.elo_a + hfa) / 400))
            y = (m.home_score > m.away_score).astype(float)
            ll = float(-(y * np.log(ph) + (1 - y) * np.log(1 - ph)).mean())
            if ll < best_ll:
                best, best_ll = p, ll
    return best
