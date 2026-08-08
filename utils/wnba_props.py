"""Generate WNBA prop picks: combine live odds + trained WNBA models.

For each (player, stat) where we have both live odds and a model, produce:
- projected_value  (model prediction)
- line             (sportsbook)
- edge             (projection - line)
- hit_prob         (historical rate of clearing the line in L20 games)
- ev               (expected value on $1 bet, using implied probability)
- pick             ("OVER" | "UNDER") based on edge sign
- confidence       ("HIGH" | "MED" | "LOW")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WnbaProp:
    player_name: str
    team: str
    stat: str
    line: float
    projected: float
    edge: float
    pick: str                       # "OVER" | "UNDER"
    hit_prob: float                 # 0.0-1.0
    ev: float                       # expected value on $1 bet
    over_price: int
    under_price: int
    bookmaker: str
    confidence: str = "LOW"         # HIGH | MED | LOW
    reasoning: list[str] = field(default_factory=list)


def american_to_implied_prob(price: Optional[int]) -> float:
    if price is None:
        return 0.5
    p = float(price)
    if p > 0:
        return 100.0 / (p + 100.0)
    return abs(p) / (abs(p) + 100.0)


def american_to_decimal(price: Optional[int]) -> float:
    if price is None:
        return 1.91  # -110 roughly
    p = float(price)
    if p > 0:
        return 1.0 + p / 100.0
    return 1.0 + 100.0 / abs(p)


def _confidence(edge_abs: float, hit_prob: float) -> str:
    """Simple heuristic: big edge + high historical rate = HIGH."""
    if edge_abs >= 2.5 and hit_prob >= 0.65:
        return "HIGH"
    if edge_abs >= 1.5 and hit_prob >= 0.55:
        return "MED"
    return "LOW"


# Combo stat -> list of component stats to sum
_COMBO_COMPONENTS = {
    "PTS+REB": ["PTS", "REB"],
    "PTS+AST": ["PTS", "AST"],
    "REB+AST": ["REB", "AST"],
    "PTS+REB+AST": ["PTS", "REB", "AST"],
}


def _clip_prediction(stat: str, value: float) -> float:
    """Clip predictions to physically-plausible ranges.

    Stats can never be negative. Cap upper bounds at reasonable WNBA maxes
    (a defense against out-of-distribution feature vectors producing garbage).
    """
    if value is None:
        return 0.0
    v = max(0.0, float(value))
    caps = {"PTS": 55.0, "REB": 22.0, "AST": 15.0, "FG3M": 10.0}
    return min(v, caps.get(stat, v))


def _project_stat(stat: str, feat_row: dict, predictor_getter) -> float | None:
    """Return model projection for a stat (single or combo). None if unavailable."""
    components = _COMBO_COMPONENTS.get(stat, [stat])
    total = 0.0
    for c in components:
        model = predictor_getter(c)
        if model is None:
            return None
        try:
            pred = model.predict(feat_row)
            raw = pred.get("predicted_value", pred) if isinstance(pred, dict) else pred
            total += _clip_prediction(c, float(raw))
        except Exception:
            return None
    return total


def _actual_stat_sum(recent: pd.DataFrame, stat: str) -> pd.Series:
    """Return per-game actual value for a stat (single or combo)."""
    components = _COMBO_COMPONENTS.get(stat, [stat])
    result = pd.Series(0.0, index=recent.index)
    for c in components:
        if c not in recent.columns:
            return pd.Series(dtype=float)  # empty → caller treats as unavailable
        result = result + recent[c].fillna(0)
    return result


# Minimum L20 average for a synthetic prop to be considered "meaningful"
# (below this, the prop is on a bench player and not worth surfacing).
_SYNTHETIC_MIN_LINE = {
    "PTS": 6.5, "REB": 2.5, "AST": 2.5, "FG3M": 1.5,
    "PTS+REB": 10.5, "PTS+AST": 9.5, "REB+AST": 5.5, "PTS+REB+AST": 12.5,
}


def _synthetic_line_from_recent(recent: pd.DataFrame, stat: str,
                                lookback: int = 20,
                                min_meaningful: bool = True) -> Optional[float]:
    """Build a half-point-rounded line from the player's last `lookback` games.

    Uses the MEDIAN (not the mean) as the center, so the historical OVER/UNDER
    split is naturally ~50/50 rather than the upward-biased mean-rounding that
    caused every synthetic pick to be an UNDER.

    For integer-rounded medians we jitter by 0.5 based on recent trend (L5 vs
    L20) — rising trend nudges the line DOWN (a real starter on a hot streak
    should have to clear a lower bar to say OVER), falling trend nudges UP.

    When `min_meaningful=True`, returns None for stats where the L20 median is
    below the "worth-a-bet" floor (bench-player noise gets filtered).
    """
    if len(recent) < 10:
        # Need at least 10 games in the window for a stable estimate
        return None
    actuals = _actual_stat_sum(recent.head(lookback), stat)
    if actuals.empty:
        return None

    center = float(actuals.median())
    if min_meaningful and center < _SYNTHETIC_MIN_LINE.get(stat, 0):
        return None

    line = round(center * 2) / 2
    if line == int(line):
        l5 = _actual_stat_sum(recent.head(5), stat)
        recent_avg = float(l5.mean()) if not l5.empty else center
        # Rising form (L5 > median) → nudge DOWN to give OVER a fair shot;
        # falling form → nudge UP.
        line = line - 0.5 if recent_avg >= center else line + 0.5
    return max(0.5, line)


def _blend_projection(
    model_pred: float,
    l20_avg: float,
    stat: str,
    alpha: float = 0.65,
) -> float:
    """Blend the model's context-adjusted projection with the player's recent form.

    Prevents catastrophic under-projections (e.g. "1.0 REB for a 4.5 avg
    player") even if the model temporarily produces an out-of-distribution
    output. The clip bounds guarantee the projection stays within a reasonable
    band around what the player has actually been doing lately.
    """
    if l20_avg <= 0:
        return _clip_prediction(stat, model_pred)
    blended = alpha * model_pred + (1 - alpha) * l20_avg
    # Never project below 40% or above 160% of the player's recent form
    lower = 0.4 * l20_avg
    upper = 1.6 * l20_avg
    return _clip_prediction(stat, max(lower, min(upper, blended)))


SYNTHETIC_BOOKMAKER = "L20 avg"


def _build_prop(
    *, player_name: str, team: str, stat: str, line: float, projected: float,
    actual_series: pd.Series, over_price: int, under_price: int,
    bookmaker: str, recent_n: int,
) -> WnbaProp:
    """Compose a WnbaProp from a projection + a line."""
    edge = projected - line
    pick = "OVER" if edge > 0 else "UNDER"
    if pick == "OVER":
        hit_prob = float((actual_series > line).mean())
        price = over_price
    else:
        hit_prob = float((actual_series < line).mean())
        price = under_price
    decimal = american_to_decimal(price)
    ev = hit_prob * (decimal - 1) - (1 - hit_prob)

    reasoning = []
    if abs(edge) >= 2:
        reasoning.append(f"Projection {projected:.1f} vs line {line:.1f} ({edge:+.1f})")
    if hit_prob >= 0.7:
        reasoning.append(f"Cleared {int(hit_prob*100)}% of last {recent_n} games")
    elif hit_prob <= 0.3:
        reasoning.append(f"Only cleared {int(hit_prob*100)}% of last {recent_n} games (fade signal)")

    return WnbaProp(
        player_name=player_name, team=team, stat=stat, line=line,
        projected=projected, edge=edge, pick=pick, hit_prob=hit_prob, ev=ev,
        over_price=over_price or -110, under_price=under_price or -110,
        bookmaker=bookmaker,
        confidence=_confidence(abs(edge), hit_prob),
        reasoning=reasoning,
    )


def generate_wnba_props(
    wnba_df: pd.DataFrame,
    predictor_getter,           # callable: stat_upper -> StatPredictor|None
    odds: dict,                 # {player: {stat: {line, over_price, under_price, bookmaker}}}
    min_recent_games: int = 5,
    todays_games: list | None = None,   # from wnba_data_fetch.get_todays_wnba_games()
    team_def: pd.DataFrame | None = None,
    def_vs_pos: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
    only_active_tonight: bool = True,
    synthesize_missing: bool = True,
    min_avg_min: float = 15.0,
    exclude_injured: bool = True,
) -> list[WnbaProp]:
    """Build list of WnbaProp objects.

    Real sportsbook odds win when available. When `synthesize_missing=True`
    (default), any (player, stat) combo that has no live odds gets a
    self-generated line from the player's last-20-game median — so the props
    board still populates when the Odds API is unreachable or over quota.

    `min_avg_min` filters synthetic props to players whose L20 avg minutes
    meets the threshold (default 15). Bench players are excluded — starters
    and rotational players only. Does NOT filter players with real sportsbook
    odds (if a book took a line on them, they're worth analyzing).

    Model projections are blended with each player's L20 avg (65% model + 35%
    recent form) and clipped to [0.4×, 1.6×] of that average — anchors the
    projection so extreme model outputs don't produce absurd picks.

    Handles single stats (PTS/AST/REB/FG3M) and combos (PTS+REB, PTS+AST,
    REB+AST, PTS+REB+AST).
    """
    if wnba_df.empty:
        return []
    odds = odds or {}

    stats_available = ["PTS", "AST", "REB", "FG3M",
                       "PTS+REB", "PTS+AST", "REB+AST", "PTS+REB+AST"]

    df_sorted = wnba_df.sort_values(["PLAYER_NAME", "_date"], ascending=[True, False])
    recent_by_player = {
        name: group.head(20) for name, group in df_sorted.groupby("PLAYER_NAME")
    }

    tonight_teams: set[str] = set()
    if todays_games:
        for g in todays_games:
            for side in ("home", "away"):
                abbrev = g.get(side, {}).get("abbrev", "")
                if abbrev:
                    tonight_teams.add(abbrev)

    from utils.wnba_predict import build_tonight_feature_row, get_tonight_matchup_for_player

    # Determine which players to score. Union of players with odds + all tonight-active players.
    candidate_players: set[str] = set(odds.keys())
    if synthesize_missing and tonight_teams:
        for name, group in recent_by_player.items():
            team = group.iloc[0].get("TEAM_ABBREVIATION", "")
            if team in tonight_teams:
                candidate_players.add(name)

    # Filter out injured players (OUT / OUT_SEASON / DOUBTFUL) — they aren't
    # playing tonight, so any prop on them is dead money.
    if exclude_injured:
        try:
            from utils.wnba_injuries import is_player_unavailable
            candidate_players = {n for n in candidate_players if not is_player_unavailable(n)}
        except Exception as e:
            logger.debug(f"[WNBA-Props] injury filter skipped: {e}")

    props: list[WnbaProp] = []

    for player_name in candidate_players:
        recent = recent_by_player.get(player_name)
        if recent is None or len(recent) < min_recent_games:
            continue
        team = recent.iloc[0].get("TEAM_ABBREVIATION", "—")

        if only_active_tonight and tonight_teams and team not in tonight_teams:
            continue

        # Only exclude bench players from the SYNTHETIC pool. If a sportsbook
        # took a line on the player, they're at least rotationally relevant.
        has_real_odds = player_name in odds
        avg_min = float(recent["MIN"].mean()) if "MIN" in recent.columns else 0.0
        if not has_real_odds and avg_min < min_avg_min:
            continue

        # Build feature row for tonight when possible
        feat_row = None
        if todays_games:
            matchup = get_tonight_matchup_for_player(team, todays_games)
            if matchup is not None:
                opp, is_home = matchup
                feat_row = build_tonight_feature_row(
                    recent, tonight_opponent=opp, is_home=is_home,
                    team_def=team_def, def_vs_pos=def_vs_pos, team_stats=team_stats,
                )
        if feat_row is None:
            feat_row = recent.iloc[0].to_dict()

        player_odds = odds.get(player_name, {})

        for stat in stats_available:
            entry = player_odds.get(stat)
            raw_projected = _project_stat(stat, feat_row, predictor_getter)
            if raw_projected is None:
                continue
            actual_series = _actual_stat_sum(recent, stat)
            if actual_series.empty:
                continue

            # Blend model output with recent form (safety net vs extreme outputs)
            l20_avg = float(actual_series.mean())
            projected = _blend_projection(raw_projected, l20_avg, stat)

            if entry is not None:
                # Real sportsbook line
                props.append(_build_prop(
                    player_name=player_name, team=team, stat=stat,
                    line=float(entry["line"]), projected=projected,
                    actual_series=actual_series,
                    over_price=entry.get("over_price") or -110,
                    under_price=entry.get("under_price") or -110,
                    bookmaker=entry.get("bookmaker", "—"),
                    recent_n=len(recent),
                ))
            elif synthesize_missing:
                # Synthetic line from L20 median (unbiased)
                line = _synthetic_line_from_recent(recent, stat)
                if line is None:
                    continue
                props.append(_build_prop(
                    player_name=player_name, team=team, stat=stat,
                    line=line, projected=projected,
                    actual_series=actual_series,
                    over_price=-110, under_price=-110,
                    bookmaker=SYNTHETIC_BOOKMAKER,
                    recent_n=len(recent),
                ))

    props.sort(key=lambda p: p.ev, reverse=True)
    return props


def props_by_stat(props: list[WnbaProp]) -> dict[str, list[WnbaProp]]:
    """Group props by stat, each list already sorted by EV."""
    out: dict[str, list[WnbaProp]] = {}
    for p in props:
        out.setdefault(p.stat, []).append(p)
    return out
