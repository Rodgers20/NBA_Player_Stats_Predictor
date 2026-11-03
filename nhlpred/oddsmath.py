"""Probability <-> American odds, and count-distribution tails (Poisson / negative binomial)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

MIN_P, MAX_P = 0.005, 0.995


def fair_american(p: float | None) -> int | None:
    """No-vig American odds for probability p: 0.6 -> -150, 0.4 -> +150."""
    if p is None or not 0 < p < 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def american_to_prob(odds: float | None) -> float | None:
    if odds is None or odds == 0 or (isinstance(odds, float) and math.isnan(odds)):
        return None
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def p_at_least(mu: np.ndarray | float, k: int, alpha: float = 0.0) -> np.ndarray:
    """P(X >= k) for a count with mean mu and variance mu + alpha*mu^2 (alpha 0 -> Poisson)."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-6)
    if alpha <= 1e-6:
        return stats.poisson.sf(k - 1, mu)
    n = 1.0 / alpha
    return stats.nbinom.sf(k - 1, n, n / (n + mu))


def line_to_k(line: float) -> int:
    """Over 2.5 means at least 3."""
    return int(math.floor(line)) + 1
