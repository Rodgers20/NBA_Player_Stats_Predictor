"""Calibrated joint-probability math for parlays.

Background
----------
Parlays claiming 90-100% win probability actually won 44.4% (8/18) over the
2026-03..2026-05 tracked history. Two compounding defects caused it:

1. ``parlay_odds()`` recovered each leg's probability from its *American odds*,
   which ``_prob_to_american()`` had already inflated by the 4.76% sportsbook
   vig. Nothing removed that vig again, so every leg was overstated by ~3-4
   points and the error compounded as ``1.0476 ** n_legs``. Worse, the clamp
   inside ``_prob_to_american()`` maps everything at or above 0.945 to -9899,
   which reads back as exactly 0.99 — so 0.95 and 0.99 became indistinguishable.
   Three such legs multiply to 0.97, which is precisely the 97.0% / -3266 seen
   on every one of the 18 overconfident slips.

2. Per-leg probabilities came from a Normal CDF whose ``std_dev`` was the
   player's last-10-game standard deviation. That captures a player's spread
   around their *recent mean* but excludes model prediction error, so the
   predictive distribution was far too tight and produced 0.99s that the data
   never supports.

Empirical constants below were measured on 2026-09-10 from 1,801 graded props
and 146 resolved parlays, excluding the mis-graded 2026-04-24 slate.
"""

from __future__ import annotations

import math

# Sportsbook vig baked in by ``parlay_builder._prob_to_american``.
VIG = 0.0476

# Residual SD of the actual outcome around the projection, by stat.
# Measured over 1,801 graded props (2026-04-06 .. 2026-05-28).
# These already include BOTH player game-to-game variance and model error,
# which is exactly the predictive spread a leg probability needs.
EMPIRICAL_RESID_SD: dict[str, float] = {
    "PTS":         7.38,
    "REB":         3.12,
    "AST":         2.65,
    "FG3M":        1.48,
    "STL":         1.04,
    "BLK":         1.06,
    "PTS+REB":     8.30,
    "PTS+AST":     8.46,
    "AST+REB":     3.95,
    "PTS+AST+REB": 9.55,
}
DEFAULT_RESID_SD = 5.67  # pooled across all 1,801 graded props

# Highest hit rate observed in any edge bucket was 86.0% (avg-line >= 10,
# n=43). 0.90 leaves headroom above that while blocking the 0.99s.
MAX_LEG_PROB = 0.90
MIN_LEG_PROB = 0.10

# Phi-correlation between two legs on the SAME player: +0.196 (n=52).
# Across DIFFERENT players: -0.002 (n=1436) — statistically indistinguishable
# from independence, so no cross-player term is modelled.
SAME_PLAYER_RHO = 0.196

# Shrinkage toward the observed marginal leg hit rate, used only when a leg
# carries no projection/line and cannot be recomputed from scratch.
BASE_LEG_RATE = 0.6443
_SHRINK = 0.65

# Gauss-Hermite nodes for the one-factor copula integral.
_GH_NODES = 48


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def devig_american(american: int) -> float:
    """Recover the TRUE probability from odds produced by _prob_to_american.

    ``_prob_to_american`` multiplies the true probability by ``1 + VIG`` before
    converting. This is its inverse. Reading the raw implied probability
    without dividing the vig back out is the original bug.
    """
    american = int(american)
    if american < 0:
        implied = abs(american) / (abs(american) + 100.0)
    else:
        implied = 100.0 / (american + 100.0)
    return min(implied / (1.0 + VIG), 1.0)


def resid_sd(stat: str) -> float:
    """Empirical predictive SD for a stat, falling back to the pooled value."""
    return EMPIRICAL_RESID_SD.get((stat or "").upper(), DEFAULT_RESID_SD)


def calibrate_leg_prob(
    prob: float,
    stat: str = "",
    projection: float | None = None,
    line: float | None = None,
    direction: str = "Over",
) -> float:
    """Return a leg probability the tracked outcomes actually support.

    When ``projection`` and ``line`` are both known the probability is
    recomputed from scratch against the empirically measured residual SD for
    that stat, discarding the too-tight sigma the props pipeline used. When
    they are not, the supplied probability is shrunk toward the observed
    marginal leg hit rate. Either way the result is clamped to
    [MIN_LEG_PROB, MAX_LEG_PROB].

    Note this is deliberately a monotone transform of the input, so it never
    reorders legs — prop *ranking* (which the props pipeline is measurably
    good at) is preserved; only the absolute values parlays multiply change.
    """
    if projection is not None and line is not None:
        sd = resid_sd(stat)
        z = (float(projection) - float(line)) / sd
        if str(direction).lower().startswith("u"):
            z = -z
        calibrated = _norm_cdf(z)
    else:
        p = min(max(float(prob), 0.0), 1.0)
        calibrated = BASE_LEG_RATE + _SHRINK * (p - BASE_LEG_RATE)

    return min(max(calibrated, MIN_LEG_PROB), MAX_LEG_PROB)


def _joint_correlated(probs: list[float], rho: float) -> float:
    """P(all hit) for equicorrelated legs under a one-factor Gaussian copula.

        P = INT phi(m) PROD_i Phi((z_i - sqrt(rho) m) / sqrt(1 - rho)) dm

    Reduces exactly to the independent product when rho == 0.
    """
    if not probs:
        return 0.0
    if rho <= 0.0:
        out = 1.0
        for p in probs:
            out *= p
        return out

    zs = [_norm_ppf(p) for p in probs]
    a = math.sqrt(rho)
    b = math.sqrt(1.0 - rho)

    # Gauss-Legendre over [-6, 6] against the standard normal density.
    lo, hi = -6.0, 6.0
    n = _GH_NODES
    total = 0.0
    for i in range(n):
        # midpoint rule is ample at n=48 over a 12-sigma span
        m = lo + (hi - lo) * (i + 0.5) / n
        w = math.exp(-0.5 * m * m) / math.sqrt(2.0 * math.pi)
        inner = 1.0
        for z in zs:
            inner *= _norm_cdf((z - a * m) / b)
        total += w * inner
    total *= (hi - lo) / n
    return min(max(total, 0.0), min(probs))


def joint_probability(
    legs: list[dict],
    same_player_rho: float = SAME_PLAYER_RHO,
) -> float:
    """Joint win probability for a parlay.

    Legs on the same player are positively correlated (rho = +0.196) and are
    combined with a one-factor Gaussian copula. Legs on different players are
    independent (measured rho = -0.002) and simply multiply.

    Each leg needs ``true_prob`` (preferred) or ``model_odds``, plus ``player``.
    """
    if not legs:
        return 0.0

    groups: dict[str, list[float]] = {}
    for i, leg in enumerate(legs):
        p = leg.get("true_prob")
        if p is None:
            p = devig_american(leg.get("model_odds", -110))
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        # Legs without a player name cannot be correlated with anything.
        key = (leg.get("player") or f"__leg_{i}__").lower()
        groups.setdefault(key, []).append(p)

    out = 1.0
    for probs in groups.values():
        out *= _joint_correlated(probs, same_player_rho) if len(probs) > 1 else probs[0]
    return min(max(out, 0.0), 1.0)
