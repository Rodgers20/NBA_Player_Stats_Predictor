"""American-odds conversions."""

from __future__ import annotations

import math


def american_to_prob(odds: float | None) -> float | None:
    """Implied probability, vig included: -130 -> 0.565, +150 -> 0.4."""
    if odds is None or math.isnan(odds) or odds == 0:
        return None
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def devig_two_way(odds_a: float | None, odds_b: float | None) -> tuple[float, float] | None:
    """Vig-free probabilities of a two-outcome market (multiplicative method)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    if pa is None or pb is None:
        return None
    return pa / (pa + pb), pb / (pa + pb)


def parse_american(text: str | float | int | None) -> float | None:
    """'+245' / '-110' / 'EVEN' / 245 -> float, None for anything else."""
    if text is None or text == "":
        return None
    if str(text).upper() == "EVEN":
        return 100.0
    try:
        return float(str(text).replace("+", ""))
    except ValueError:
        return None
