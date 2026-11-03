"""Price every eligible player of a fixture: anytime goal, assist, shots on target over 0.5 / 1.5.

The same function serves the backtest (team expectations from a simple opponent-concession proxy)
and production (team expectations from the match model), so what is validated is what ships.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from . import model
from .forms import PlayerForm, TeamAggregate

MARKETS = ("anytime_goal", "assist", "shots_on_target")
TEAM_PRIOR_GAMES = 4.0  # team scoring / shooting rates are shrunk toward the league mean by this many games
DEFAULT_DISPERSION = {"goals": 0.0, "assists": 0.0, "sot": 0.0}


def position_priors(forms: Iterable[PlayerForm]) -> dict[str, dict[str, float]]:
    """Per-full-game goal / assist / SOT rate by outfield position, from everyone's totals."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"goals": 0.0, "assists": 0.0, "sot": 0.0, "exposure": 0.0})
    for f in forms:
        if f.position in model.GATED_POSITIONS:
            continue
        for key in ((f.position,) if f.position else ()) + ("ALL",):  # never-started subs have no position: ALL only
            t = totals[key]
            t["goals"] += f.goals
            t["assists"] += f.assists
            t["sot"] += f.sot
            t["exposure"] += model.exposure(f.starts, f.subs)
    return {
        pos: {k: (t[k] / t["exposure"] if t["exposure"] else 0.0) for k in ("goals", "assists", "sot")}
        for pos, t in totals.items()
    }


def _shrunk_per_game(total: float, games: int, league_avg: float) -> float:
    return (total + TEAM_PRIOR_GAMES * league_avg) / (games + TEAM_PRIOR_GAMES)


def proxy_expected(attack: TeamAggregate | None, defence: TeamAggregate | None, league_avg: float, field: str) -> float:
    """Opponent-concession proxy: league_avg * (team attack / league) * (opponent defence / league)."""
    if not league_avg:
        return 0.0
    f = "goals_for" if field == "goals" else "sot_for"
    a = "goals_against" if field == "goals" else "sot_against"
    att = _shrunk_per_game(getattr(attack, f, 0), getattr(attack, "games", 0), league_avg) / league_avg
    deff = _shrunk_per_game(getattr(defence, a, 0), getattr(defence, "games", 0), league_avg) / league_avg
    return league_avg * att * deff


def _factor(expected: float, agg: TeamAggregate | None, field: str, league_avg: float) -> float:
    """How far this match's team expectation sits above the team's own (shrunk) season average."""
    if agg is None or not league_avg or not expected:
        return 1.0
    total = agg.goals_for if field == "goals" else agg.sot_for
    base = _shrunk_per_game(total, agg.games, league_avg)
    return expected / base if base else 1.0


def _reason(market: str, f: PlayerForm, per_game: float, opp: str, opp_agg: TeamAggregate | None, xg: float) -> str:
    if market == "shots_on_target":
        against = f", {opp} allow {opp_agg.sot_against / opp_agg.games:.1f} SOT/game" if opp_agg and opp_agg.games else ""
        return f"{per_game:.2f} SOT per appearance over {f.games}{against}"
    noun = "goals" if market == "anytime_goal" else "assists"
    against = f", {opp} concede {opp_agg.goals_against / opp_agg.games:.1f}/game" if opp_agg and opp_agg.games else ""
    return f"{per_game:.2f} {noun} per appearance over {f.games}; {f.team} expected {xg:.1f} goals{against}"


def _price_side(forms: list[PlayerForm], team: str, opp: str, is_home: bool, aggs: dict, priors: dict, league: dict,
                xg: float, sot: float, markets: tuple[str, ...], phi: dict, strength: dict) -> list[dict]:
    rows: list[dict] = []
    f_goal = _factor(xg, aggs.get(team), "goals", league["goals"])
    f_sot = _factor(sot, aggs.get(team), "sot", league["sot"])
    opp_agg = aggs.get(opp)
    for f in (x for x in forms if x.team == team):
        sub_rate = f.subs / max(1, f.team_games - f.starts)
        start_p = model.start_probability(f.starts, f.team_games, f.last5_starts, f.last5_games, f.gap)
        if not model.eligible(f.games, start_p, f.position):
            continue
        prior = priors.get(f.position or "ALL") or priors.get("ALL") or {"goals": 0.0, "assists": 0.0, "sot": 0.0}
        x = model.expected_exposure(start_p, sub_rate)
        exp_full = model.exposure(f.starts, f.subs)
        base = {"espn_id": f.espn_id, "player": f.player, "team": team, "opponent": opp, "is_home": is_home,
                "position": f.position, "games": f.games, "start_prob": round(start_p, 4)}
        specs = (
            ("anytime_goal", "goals", f.goals, f_goal, [(None, 1)]),
            ("assist", "assists", f.assists, f_goal, [(None, 1)]),
            ("shots_on_target", "sot", f.sot, f_sot, [(0.5, 1), (1.5, 2)]),
        )
        for market, key, total, factor, lines in specs:
            if market not in markets:
                continue
            rate = model.shrunk_rate(total, exp_full, prior[key], strength[key])
            mu = rate * x * factor
            per_game = total / f.games
            for line, k in lines:
                rows.append({**base, "market": market, "line": line,
                             "prob": round(model.at_least(mu, k, phi[key]), 4), "expected": round(mu, 4),
                             "per_game": round(per_game, 4), "confidence": model.confidence(f.games, market),
                             "reason": _reason(market, f, per_game, opp, opp_agg, xg)})
    return rows


def price_fixture(forms: list[PlayerForm], aggs: dict[str, TeamAggregate], priors: dict, league: dict[str, float],
                  home: str, away: str, expected_goals: dict[str, float], expected_sot: dict[str, float] | None,
                  markets: tuple[str, ...] = MARKETS, dispersion: dict[str, float] | None = None,
                  strength: dict[str, float] | None = None) -> list[dict]:
    """All prop rows for one fixture, sorted by probability within each (market, line)."""
    phi = {**DEFAULT_DISPERSION, **(dispersion or {})}
    strengths = {**model.PRIOR_STRENGTH, **(strength or {})}
    sot = expected_sot or {
        "home": proxy_expected(aggs.get(home), aggs.get(away), league["sot"], "sot"),
        "away": proxy_expected(aggs.get(away), aggs.get(home), league["sot"], "sot"),
    }
    rows = (
        _price_side(forms, home, away, True, aggs, priors, league, expected_goals["home"], sot["home"], markets, phi, strengths)
        + _price_side(forms, away, home, False, aggs, priors, league, expected_goals["away"], sot["away"], markets, phi, strengths)
    )
    return sorted(rows, key=lambda r: (r["market"], r["line"] or 0, -r["prob"]))
