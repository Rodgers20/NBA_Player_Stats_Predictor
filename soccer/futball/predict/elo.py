"""Causal Elo ratings.

Elo is the strongest single classical feature in football forecasting, and it is
cheap: one pass over date-sorted matches, recording each team's rating *before*
the match and updating it after. The pre-match rating is what becomes a feature,
so there is no leakage by construction.

Two refinements over textbook Elo, both standard in football work:

* **Home advantage** is added to the home side's rating when computing the
  expected score, rather than being left for the model to infer.
* **The K factor scales with goal difference**, so a 4-0 moves ratings further
  than a 1-0. Without this, Elo treats a thrashing and a scrape identically.

Ratings are keyed on team NAME. Sources spell clubs differently ("Bayern Munich"
in the Bundesliga file, "FC Bayern München" in the Champions League file), so
this naturally keeps competition-specific ratings apart rather than silently
merging two different-looking entities.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import polars as pl

START_RATING = 1500.0
HOME_ADVANTAGE = 65.0
BASE_K = 20.0


@dataclass
class EloConfig:
    start: float = START_RATING
    home_advantage: float = HOME_ADVANTAGE
    k: float = BASE_K


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _goal_multiplier(goal_diff: int, rating_diff: float) -> float:
    """Dampens runaway updates when a strong side thrashes a weak one."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    return (gd + 1.0) ** 0.6 / (1.0 + 0.001 * abs(rating_diff)) ** 0.3


def attach(df: pl.DataFrame, cfg: EloConfig | None = None) -> pl.DataFrame:
    """Add pre-match Elo columns to a date-sorted results frame."""
    cfg = cfg or EloConfig()
    rows = df.sort(["date", "match_key"]).to_dicts()

    ratings: dict[str, float] = {}
    home_elo: list[float] = []
    away_elo: list[float] = []
    elo_diff: list[float] = []
    home_exp: list[float] = []

    for _, batch in groupby(rows, key=lambda r: r["date"]):
        updates: dict[str, float] = {}
        for r in batch:
            h, a = r["home_team"], r["away_team"]
            rh = ratings.get(h, cfg.start)
            ra = ratings.get(a, cfg.start)
    
            # Record the ratings as they stood BEFORE this match.
            home_elo.append(rh)
            away_elo.append(ra)
            elo_diff.append(rh - ra)
            exp_h = _expected(rh + cfg.home_advantage, ra)
            home_exp.append(exp_h)
    
            hg, ag = r["home_goals"], r["away_goals"]
            if hg is None or ag is None:
                continue
    
            score_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            mult = _goal_multiplier(hg - ag, rh - ra)
            delta = cfg.k * mult * (score_h - exp_h)
    
            updates[h] = updates.get(h, 0.0) + delta
            updates[a] = updates.get(a, 0.0) - delta
        for team, delta in updates.items():
            ratings[team] = ratings.get(team, cfg.start) + delta

    return (
        df.sort(["date", "match_key"])
        .with_columns(
            pl.Series("home_elo", home_elo, dtype=pl.Float64),
            pl.Series("away_elo", away_elo, dtype=pl.Float64),
            pl.Series("elo_diff", elo_diff, dtype=pl.Float64),
            pl.Series("elo_expected_home", home_exp, dtype=pl.Float64),
        )
    )


def final_ratings(df: pl.DataFrame, cfg: EloConfig | None = None) -> dict[str, float]:
    """Ratings after the last match in `df` — used at serve time."""
    cfg = cfg or EloConfig()
    ratings: dict[str, float] = {}
    for r in df.sort(["date", "match_key"]).to_dicts():
        h, a = r["home_team"], r["away_team"]
        hg, ag = r["home_goals"], r["away_goals"]
        if hg is None or ag is None:
            continue
        rh = ratings.get(h, cfg.start)
        ra = ratings.get(a, cfg.start)
        exp_h = _expected(rh + cfg.home_advantage, ra)
        score_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        delta = cfg.k * _goal_multiplier(hg - ag, rh - ra) * (score_h - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    return ratings
