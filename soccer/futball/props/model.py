"""Player prop maths: shrunk per-game rates, expected exposure, and count-tail probabilities.

Pure functions, no I/O. The pipeline is

    rate      = (stat_total + k * position_prior) / (exposure + k)        empirical-Bayes shrinkage
    mu        = rate * expected_exposure * team_factor                    expected count this match
    P(X >= n) = Poisson, or negative binomial when `phi` > 0              count tail

`exposure` is measured in full-game equivalents (a start is 1, a substitute appearance 0.3)
because the free ESPN box scores carry no minutes. `team_factor` rescales a player's rate by how
many goals / shots on target his team is expected to produce in THIS match relative to its season
average, so a strong opponent or a home game moves every teammate together.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SUB_WEIGHT = 0.3  # a substitute appearance is worth ~25 of 90 minutes
MIN_GAMES = 2
MIN_START_PROB = 0.25
GATED_POSITIONS = ("G",)

#: Prior strength in full-game equivalents: how many games of position-average evidence a player's own
#: record is blended with. Goals and assists are sparse, so they get more shrinkage than shots on target.
PRIOR_STRENGTH = {"goals": 6.0, "assists": 8.0, "sot": 3.0}


def at_least(mu: float, k: int, phi: float = 0.0) -> float:
    """P(X >= k) for k in {1, 2}; Poisson, or negative binomial (variance mu + phi*mu^2) if phi > 0."""
    if k not in (1, 2):
        raise ValueError("only k=1 and k=2 are supported")
    if mu <= 0:
        return 0.0
    if phi <= 1e-6:
        p0 = math.exp(-mu)
        p1 = p0 * mu
    else:
        r = 1.0 / phi
        p0 = (r / (r + mu)) ** r
        p1 = p0 * r * (mu / (r + mu))
    return 1.0 - p0 if k == 1 else 1.0 - p0 - p1


def shrunk_rate(total: float, exposure: float, prior_rate: float, strength: float) -> float:
    """Per-full-game rate pulled toward `prior_rate` with the weight of `strength` games."""
    return (total + strength * prior_rate) / (exposure + strength)


def exposure(starts: int, subs: int) -> float:
    """Full-game equivalents played: starts count 1, substitute appearances SUB_WEIGHT."""
    return starts + SUB_WEIGHT * subs


def start_probability(starts: int, team_games: int, last5_starts: int, last5_games: int, gap: int) -> float:
    """Chance he starts the next match: 60% last five team games, 40% season, 0.3x if absent 3+ matches."""
    season = starts / team_games if team_games else 0.0
    recent = last5_starts / last5_games if last5_games else season
    blended = 0.6 * recent + 0.4 * season
    if gap >= 3:  # not in the matchday squad for 3+ team matches: injured, dropped or loaned
        blended *= 0.3
    return max(0.0, min(1.0, blended))


def expected_exposure(start_p: float, sub_rate: float) -> float:
    """Expected full-game equivalents next match: a starter plays 1, otherwise sub_rate * SUB_WEIGHT."""
    return start_p + (1.0 - start_p) * max(0.0, min(1.0, sub_rate)) * SUB_WEIGHT


def confidence(games: int, market: str) -> str:
    """high >= 8 games, medium 4-7, low < 4. Assists are the thinnest data: low until 8 games, then medium."""
    if market == "assist":
        return "medium" if games >= 8 else "low"
    return "high" if games >= 8 else "medium" if games >= 4 else "low"


def eligible(games: int, start_p: float, position: str | None) -> bool:
    """Who gets a prop row at all: 2+ appearances, a real chance of playing, and not a goalkeeper."""
    return games >= MIN_GAMES and start_p >= MIN_START_PROB and position not in GATED_POSITIONS


PARAMS_FILE = Path(__file__).with_name("params.json")


def load_params() -> dict:
    """Tuned {"strength": {...}, "dispersion": {...}} from the last backtest; defaults if none was saved."""
    try:
        return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"strength": dict(PRIOR_STRENGTH), "dispersion": {"sot": 0.0}}


def save_params(params: dict) -> None:
    PARAMS_FILE.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
