"""NBA next-game projections for the player page.

Mirrors the WNBA path so both leagues show the same Next Game Prediction card:

    utils/wnba_predict.build_tonight_feature_row   -> project_next_game (here)
    utils/wnba_props._blend_projection             -> blend_projection (here)
    utils/wnba_props._clip_prediction              -> clip_prediction (here)

The feature engineering itself is shared, not duplicated:
utils.pregame_features.next_game_features is league-agnostic and already emits
exactly the 32 columns the NBA pts/ast/reb models were trained on.

Why blend at all: the models are trained on in-distribution feature vectors.
A player returning from injury, a trade, or a rotation change can produce an
out-of-distribution row and a nonsense projection (the WNBA card was showing
"1.0 REB" for a 4.5-REB player before blending was added). Blending toward
recent form and clipping to a band around it bounds that failure mode. This
card is user-facing, so a wrong-but-plausible number is far better than a
wrong-and-absurd one.

Unlike WNBA there are no per-stat calibration offsets here — data/nba/
model_calibration.json holds game-level score biases, not player-stat biases,
so nothing is subtracted. If per-stat NBA offsets are added later, apply them
to `raw` in project_next_game before blending.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Physically plausible single-game ceilings. Deliberately generous — these are
# a guard against garbage, not a forecast bound. NBA highs run well above the
# WNBA equivalents in utils/wnba_props._clip_prediction.
NBA_STAT_CAPS: dict[str, float] = {
    "PTS": 70.0,
    "REB": 30.0,
    "AST": 25.0,
    "FG3M": 14.0,
    "STL": 10.0,
    "BLK": 12.0,
}

# Model weight when blending with recent form. Matches the WNBA card.
DEFAULT_ALPHA = 0.55

# Projection may not fall below/above this multiple of the player's L20 form.
_FORM_FLOOR = 0.4
_FORM_CEIL = 1.6

_STATS = ("PTS", "AST", "REB")


def clip_prediction(stat: str, value: Optional[float]) -> float:
    """Clip a projection to a physically plausible range. Never negative."""
    if value is None:
        return 0.0
    try:
        v = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
    return min(v, NBA_STAT_CAPS.get((stat or "").upper(), v))


def blend_projection(
    model_pred: float,
    l20_avg: float,
    stat: str,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Blend the model projection with the player's recent form, then clip.

    With no usable form signal (l20_avg <= 0) the model output is returned
    clipped. Otherwise the blend is held within [0.4x, 1.6x] of L20, which is
    what stops a single bad feature vector from dominating the card.
    """
    if l20_avg is None or l20_avg <= 0:
        return clip_prediction(stat, model_pred)

    blended = alpha * float(model_pred) + (1.0 - alpha) * float(l20_avg)
    lower = _FORM_FLOOR * l20_avg
    upper = _FORM_CEIL * l20_avg
    return clip_prediction(stat, max(lower, min(upper, blended)))


def _abbrev(side) -> str:
    """Accept both the nested game shape and the flat one."""
    if isinstance(side, dict):
        return str(side.get("abbrev") or side.get("team") or "").upper()
    return str(side or "").upper()


def get_tonight_matchup_for_player(
    team_abbr: str,
    todays_games: Optional[list],
) -> Optional[tuple[str, bool]]:
    """Return (opponent_abbrev, is_home) for the team playing tonight.

    Returns None when the team is not on tonight's slate, which the caller
    uses to fall back to the player's most recent opponent.
    """
    if not team_abbr or not todays_games:
        return None

    want = str(team_abbr).upper()
    for game in todays_games:
        try:
            home = _abbrev(game.get("home"))
            away = _abbrev(game.get("away"))
        except AttributeError:
            continue
        if home == want:
            return (away, True)
        if away == want:
            return (home, False)
    return None


def _mean(df: pd.DataFrame, stat: str) -> float:
    if df.empty or stat not in df.columns:
        return 0.0
    val = pd.to_numeric(df[stat], errors="coerce").mean()
    return float(val) if pd.notna(val) else 0.0


def project_next_game(
    player_history: pd.DataFrame,
    predictor_getter: Callable[[str], object],
    opponent: str,
    is_home: bool,
    game_date: Optional[date] = None,
    stats: tuple[str, ...] = _STATS,
) -> Optional[dict]:
    """Project PTS/AST/REB for a player's next game.

    Args:
        player_history: the player's enriched rows, most recent first.
        predictor_getter: callable stat -> model (dashboard.app.get_predictor).
        opponent: opponent abbrev, used for display context.
        is_home: whether the player's team is home.

    Returns:
        {"stats": {...}, "l20": {...}, "l5": {...}, "opponent": str,
         "is_home": bool} — or None when there is no history to work from.
        A stat is omitted from "stats" when its model is missing or errors;
        the card renders whatever did resolve rather than failing whole.
    """
    if player_history is None or player_history.empty:
        return None

    from utils.pregame_features import next_game_features

    try:
        feat = next_game_features(player_history, game_date or date.today(),
                                  is_home)
    except Exception as exc:
        logger.warning("NBA next_game_features failed: %s", exc)
        return None

    l20 = player_history.head(20)
    l5 = player_history.head(5)

    projected: dict[str, float] = {}
    for stat in stats:
        model = predictor_getter(stat)
        if model is None:
            continue
        try:
            pred = model.predict(feat)
            raw = float(pred.get("predicted_value", pred)
                        if isinstance(pred, dict) else pred)
        except Exception as exc:
            logger.debug("NBA %s projection failed: %s", stat, exc)
            continue
        projected[stat] = blend_projection(raw, _mean(l20, stat), stat)

    return {
        "stats": projected,
        "l20": {s: _mean(l20, s) for s in stats},
        "l5": {s: _mean(l5, s) for s in stats},
        "opponent": opponent,
        "is_home": bool(is_home),
    }
