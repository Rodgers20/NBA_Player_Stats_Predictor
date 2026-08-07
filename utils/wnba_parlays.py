"""Build multi-leg WNBA parlays from top single-leg props.

Constraints applied:
- Only picks with positive EV
- Prefer HIGH/MED confidence legs
- Avoid two legs from the same player (correlated risk)
- Combined decimal odds = product of leg decimals (sportsbook approximation
  ignoring correlation)
- Combined hit probability = product of leg hit probabilities (independence
  assumption; real books account for correlation but sportsbooks price parlays
  favorably to the book anyway)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from utils.wnba_props import WnbaProp, american_to_decimal


@dataclass
class WnbaParlay:
    legs: list[WnbaProp]
    combined_decimal: float
    combined_american: int
    hit_prob: float          # 0-1
    ev: float                # per $1 bet


def _decimal_to_american(dec: float) -> int:
    if dec >= 2.0:
        return round((dec - 1) * 100)
    return round(-100 / (dec - 1))


def build_wnba_parlays(
    props: Iterable[WnbaProp],
    leg_counts: tuple[int, ...] = (2, 3, 4),
    max_per_size: int = 5,
    min_leg_ev: float = 0.05,
    min_leg_hit_prob: float = 0.55,
) -> list[WnbaParlay]:
    """Return top parlays across the requested leg counts, sorted by combined EV.

    - `leg_counts`: which parlay sizes to produce (default 2/3/4-leg).
    - `max_per_size`: cap on how many parlays to keep per leg count.
    - `min_leg_ev` / `min_leg_hit_prob`: filter weak individual legs before combining.
    """
    filtered = [p for p in props if p.ev >= min_leg_ev and p.hit_prob >= min_leg_hit_prob]
    # Bias toward HIGH/MED confidence for the pool
    conf_order = {"HIGH": 0, "MED": 1, "LOW": 2}
    filtered.sort(key=lambda p: (conf_order.get(p.confidence, 3), -p.ev))
    # Cap the base pool so C(n, k) doesn't explode
    pool = filtered[:20]

    all_parlays: list[WnbaParlay] = []
    for k in leg_counts:
        candidates: list[WnbaParlay] = []
        for combo in combinations(pool, k):
            # No repeated players
            names = [p.player_name for p in combo]
            if len(set(names)) != k:
                continue

            combined_decimal = 1.0
            for leg in combo:
                price = leg.over_price if leg.pick == "OVER" else leg.under_price
                combined_decimal *= american_to_decimal(price)

            combined_hit_prob = 1.0
            for leg in combo:
                combined_hit_prob *= leg.hit_prob

            ev = combined_hit_prob * (combined_decimal - 1) - (1 - combined_hit_prob)
            if ev <= 0:
                continue

            candidates.append(WnbaParlay(
                legs=list(combo),
                combined_decimal=round(combined_decimal, 3),
                combined_american=_decimal_to_american(combined_decimal),
                hit_prob=round(combined_hit_prob, 4),
                ev=round(ev, 3),
            ))

        candidates.sort(key=lambda pl: pl.ev, reverse=True)
        all_parlays.extend(candidates[:max_per_size])

    all_parlays.sort(key=lambda pl: pl.ev, reverse=True)
    return all_parlays
