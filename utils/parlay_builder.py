"""
parlay_builder.py
=================
Constructs recommended parlays from the model's best props and game predictions.

Parlays produced each cache refresh:
    1x  3-leg Moneyline parlay      (from game predictions — HIGH/MEDIUM confidence only)
    2x 10-leg 100% Alt Line parlays  (longest streaks first)
    5x  3-leg Props OVER parlays     (highest model probability)
    3x  3-leg Props UNDER parlays    (highest model probability)
    3x  3-leg Defense parlays        (BLK/STL alt lines only)

Diversity rule
--------------
Same (player, stat, direction) combo may appear in **at most 2 parlays** across
the entire output. A player can appear in more than 2 parlays if different
stat types or directions are used (e.g. LaMelo on PTS in 2 parlays, then
on AST in 2 more parlays is fine).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional


# ── Odds helpers ──────────────────────────────────────────────────────────────

def _prob_to_american(prob: float, vig: float = 0.0476) -> int:
    """True probability → American odds with sportsbook vig applied.

    Standard -110/-110 market has 4.76% vig (104.76% total implied).
    """
    prob    = max(0.01, min(0.99, prob))
    viggged = min(prob * (1 + vig), 0.99)
    if viggged >= 0.5:
        return int(-100 * viggged / (1 - viggged))
    return int(100 * (1 - viggged) / viggged)


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds back to American odds."""
    if decimal >= 2.0:
        return int((decimal - 1.0) * 100)
    return int(-100.0 / (decimal - 1.0))


def parlay_odds(legs: list[dict]) -> dict:
    """Calculate combined parlay odds from a list of bet legs.

    Each leg must have a 'model_odds' key (American integer).

    Returns:
        decimal:    Combined decimal odds
        american:   Combined American odds
        win_prob:   Joint win probability in % (product of per-leg true probs)
        payout_100: Profit on a $100 wager
    """
    if not legs:
        return {"decimal": 1.0, "american": 0, "win_prob": 0.0, "payout_100": 0.0}

    decimal_combined = 1.0
    win_prob         = 1.0

    for leg in legs:
        odds = leg.get("model_odds", -110)
        decimal_combined *= american_to_decimal(odds)
        # True (de-vigged) probability from American odds
        if odds < 0:
            win_prob *= abs(odds) / (abs(odds) + 100.0)
        else:
            win_prob *= 100.0 / (odds + 100.0)

    return {
        "decimal":    round(decimal_combined, 2),
        "american":   decimal_to_american(decimal_combined),
        "win_prob":   round(win_prob * 100, 1),
        "payout_100": round((decimal_combined - 1.0) * 100, 2),
    }


# ── Diversity tracker ─────────────────────────────────────────────────────────

class _DiversityTracker:
    """Enforces the rule: same (player, stat) pair in at most 2 parlays TOTAL.

    Direction is intentionally excluded from the key — you can't have LaMelo
    Ball's PTS in more than 2 parlays regardless of whether it's Over, Under,
    Alt, or Defense.  Different stats (e.g. PTS vs AST) are tracked separately.
    """

    def __init__(self, max_uses: int = 2) -> None:
        self._uses: dict[tuple, int] = defaultdict(int)
        self._max = max_uses

    def _key(self, player: str, stat: str, direction: str = "") -> tuple:
        # direction param accepted for interface compat but intentionally ignored
        return (player.lower(), stat.upper())

    def can_use(self, player: str, stat: str, direction: str = "") -> bool:
        return self._uses[self._key(player, stat)] < self._max

    def mark_used(self, player: str, stat: str, direction: str = "") -> None:
        self._uses[self._key(player, stat)] += 1


# ── Streak window → probability mapping ──────────────────────────────────────
# Longer streaks imply higher per-game probability.  We use the streak length
# as a confidence signal rather than computing an independent-game joint prob
# (which would underestimate probability since player performance is correlated).
_WINDOW_TO_PROB: dict[int, float] = {
    5: 0.72,  6: 0.72,  7: 0.72,
    8: 0.78, 10: 0.78, 12: 0.78,
    15: 0.84, 17: 0.84,
    18: 0.88, 20: 0.88,
}


def _streak_prob(window: int) -> float:
    """Return estimated per-game probability from alt-line streak length."""
    closest = min(_WINDOW_TO_PROB.keys(), key=lambda w: abs(w - window))
    return _WINDOW_TO_PROB[closest]


# ── Moneyline parlay ──────────────────────────────────────────────────────────

def build_ml_parlay(
    game_predictions: list[dict],
    tracker: _DiversityTracker,
) -> Optional[dict]:
    """Build a 3-leg Moneyline parlay from today's game predictions.

    Only HIGH and MEDIUM confidence picks are included.  LOW picks are too
    close to a coin flip to carry a parlay.

    Args:
        game_predictions: List of game dicts with keys:
            home, away, winner_pick, winner_confidence
        tracker: Shared diversity tracker (ML picks are game-level — tracker
                 is passed for interface consistency but ML legs don't consume
                 the player-prop budget)

    Returns:
        Parlay dict with 'legs', 'odds', 'label', 'type', or None if < 3 picks.
    """
    CONF_TO_PROB = {"HIGH": 0.78, "MEDIUM": 0.62}

    candidates: list[dict] = []
    for g in game_predictions:
        conf   = g.get("winner_confidence", "LOW")
        prob   = CONF_TO_PROB.get(conf)
        winner = g.get("winner_pick")
        if prob is None or not winner:
            continue

        home = g.get("home", "")
        away = g.get("away", "")
        # Flip probability if the away team is the predicted winner
        if winner != home:
            prob = max(0.52, 1.0 - prob)

        candidates.append({
            "player":     f"{away} @ {home}",  # used for display
            "stat":       "ML",
            "direction":  "ML",
            "pick":       winner,
            "confidence": conf,
            "win_prob":   round(prob * 100, 1),
            "model_odds": _prob_to_american(prob),
            "label":      f"{winner} ML ({conf})",
            "game":       f"{away} @ {home}",
        })

    # Best 3 by probability
    candidates.sort(key=lambda x: -x["win_prob"])
    legs = candidates[:3]

    if len(legs) < 3:
        return None

    return {
        "label": "3-Leg Moneyline Parlay",
        "type":  "ml",
        "legs":  legs,
        "odds":  parlay_odds(legs),
    }


# ── Alt line parlays (10-leg) ─────────────────────────────────────────────────

def build_alt_parlays(
    alt_lines: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 2,
    n_legs: int = 10,
) -> list[dict]:
    """Build n_parlays 10-leg parlays from 100% Alt Lines.

    Alt lines are already sorted longest→highest threshold→player name.
    We greedily fill each parlay, respecting diversity limits.

    Requires at least 8 qualifying legs to emit a parlay.
    """
    parlays: list[dict] = []

    for parlay_idx in range(n_parlays):
        legs: list[dict] = []
        used_this_parlay: set[tuple] = set()

        for alt in alt_lines:
            if len(legs) >= n_legs:
                break

            player = alt["player"]
            stat   = alt["stat"]
            key    = (player, stat)

            # No duplicate (player, stat) within the same parlay
            if key in used_this_parlay:
                continue
            # Diversity: at most 2 parlays per (player, stat, direction)
            if not tracker.can_use(player, stat, "alt"):
                continue

            prob = _streak_prob(alt["window"])
            legs.append({
                "player":     player,
                "team":       alt["team"],
                "opponent":   alt.get("opponent", ""),
                "stat":       stat,
                "stat_label": alt["stat_label"],
                "direction":  "Over",
                "threshold":  alt["threshold"],
                "window":     alt["window"],
                "trend":      alt["trend"],
                "win_prob":   round(prob * 100, 1),
                "model_odds": _prob_to_american(prob),
                "label":      f"{player} {alt['threshold']}+ {alt['stat_label']}",
                "game":       alt.get("game_matchup", ""),
            })
            used_this_parlay.add(key)

        if len(legs) >= 8:
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], "alt")
            parlays.append({
                "label": f"10-Leg Alt Lines Parlay #{parlay_idx + 1}",
                "type":  "alt",
                "legs":  legs,
                "odds":  parlay_odds(legs),
            })

    return parlays


# ── Props OVER parlays (3-leg) ────────────────────────────────────────────────

def build_over_parlays(
    props: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 5,
    n_legs: int = 3,
) -> list[dict]:
    """Build 5 three-leg OVER parlays from the highest-probability props."""
    overs = [
        p for p in props
        if p.get("direction", "").lower() == "over"
        and (p.get("model_prob") or 0) >= 0.56
        and p.get("model_over_odds") is not None
    ]
    overs.sort(key=lambda x: -(x.get("model_prob") or 0))

    return _build_prop_parlays(
        sorted_props=overs,
        tracker=tracker,
        direction="over",
        odds_key="model_over_odds",
        n_parlays=n_parlays,
        n_legs=4,       # try to fill 4 legs, emit if ≥3
        min_legs=3,
        parlay_type="over",
        label_prefix="Props OVER Parlay",
    )


# ── Props UNDER parlays (3-leg) ───────────────────────────────────────────────

def build_under_parlays(
    props: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 3,
    n_legs: int = 3,
) -> list[dict]:
    """Build 3 three-leg UNDER parlays from the highest-probability under props.

    Key: for Under props, `model_prob` IS the under probability (e.g. 0.62 means
    62% chance of going under). The correct betting odds are therefore stored in
    `model_over_odds` = _prob_to_american(model_prob). Using `model_under_odds`
    (= _prob_to_american(1 - model_prob)) would give the OVER odds for an UNDER
    prop — making each leg look like a longshot and the parlay unbeatable.
    """
    unders = [
        p for p in props
        if p.get("direction", "").lower() == "under"
        and (p.get("model_prob") or 0) >= 0.56
        and p.get("model_over_odds") is not None   # model_over_odds = _prob_to_american(model_prob)
    ]
    unders.sort(key=lambda x: -(x.get("model_prob") or 0))

    return _build_prop_parlays(
        sorted_props=unders,
        tracker=tracker,
        direction="under",
        odds_key="model_over_odds",   # correct: model_prob IS the under probability
        n_parlays=n_parlays,
        n_legs=4,       # try to fill 4 legs, emit if ≥3
        min_legs=3,
        parlay_type="under",
        label_prefix="Props UNDER Parlay",
    )


# ── Defense parlays — BLK/STL only (3-leg) ───────────────────────────────────

def build_defense_parlays(
    alt_lines: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 3,
    n_legs: int = 3,
) -> list[dict]:
    """Build 3 three-leg defense parlays using only BLK and STL alt lines."""
    defense_alts = [a for a in alt_lines if a["stat"] in ("BLK", "STL")]

    parlays: list[dict] = []
    for parlay_idx in range(n_parlays):
        legs: list[dict] = []
        used_this_parlay: set[tuple] = set()

        for alt in defense_alts:
            if len(legs) >= n_legs:
                break

            player = alt["player"]
            stat   = alt["stat"]
            key    = (player, stat)

            if key in used_this_parlay:
                continue
            if not tracker.can_use(player, stat, "def"):
                continue

            prob = _streak_prob(alt["window"])
            legs.append({
                "player":     player,
                "team":       alt["team"],
                "opponent":   alt.get("opponent", ""),
                "stat":       stat,
                "stat_label": alt["stat_label"],
                "direction":  "Over",
                "threshold":  alt["threshold"],
                "window":     alt["window"],
                "trend":      alt["trend"],
                "win_prob":   round(prob * 100, 1),
                "model_odds": _prob_to_american(prob),
                "label":      f"{player} {alt['threshold']}+ {alt['stat_label']}",
                "game":       alt.get("game_matchup", ""),
            })
            used_this_parlay.add(key)

        if len(legs) == n_legs:
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], "def")
            parlays.append({
                "label": f"Defense Parlay #{parlay_idx + 1}  (Blocks & Steals)",
                "type":  "defense",
                "legs":  legs,
                "odds":  parlay_odds(legs),
            })

    return parlays


# ── Generic prop parlay builder ───────────────────────────────────────────────

def _build_prop_parlays(
    sorted_props: list[dict],
    tracker: _DiversityTracker,
    direction: str,
    odds_key: str,
    n_parlays: int,
    n_legs: int,
    parlay_type: str,
    label_prefix: str,
    min_legs: int = 3,
) -> list[dict]:
    """Greedy builder for Over/Under prop parlays.

    Tries to fill n_legs per parlay.  Emits a parlay if at least min_legs were
    collected — so a 3-or-4-leg parlay is valid when some candidates are
    exhausted by the diversity rule.
    """
    parlays: list[dict] = []

    for parlay_idx in range(n_parlays):
        legs: list[dict] = []
        used_this_parlay: set[tuple] = set()

        for prop in sorted_props:
            if len(legs) >= n_legs:
                break

            player   = prop["player"]
            stat     = prop.get("stat_type") or prop.get("stat", "")
            key      = (player, stat)
            odds_val = prop.get(odds_key)

            if odds_val is None:
                continue
            if key in used_this_parlay:
                continue
            if not tracker.can_use(player, stat, direction):
                continue

            prob = prop.get("model_prob") or 0.56
            legs.append({
                "player":     player,
                "team":       prop.get("team", ""),
                "stat":       stat,
                "direction":  direction.capitalize(),
                "line":       prop.get("line"),
                "win_prob":   round(prob * 100, 1),
                "model_odds": odds_val,
                "hit_rate":   prop.get("hit_rate"),
                "ev":         prop.get("ev"),
                "label":      f"{player} {direction.capitalize()} {prop.get('line')} {stat}",
                "game":       prop.get("game_matchup", ""),
            })
            used_this_parlay.add(key)

        if len(legs) >= min_legs:
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], direction)
            parlays.append({
                "label": f"{len(legs)}-Leg {label_prefix} #{parlay_idx + 1}",
                "type":  parlay_type,
                "legs":  legs,
                "odds":  parlay_odds(legs),
            })

    return parlays


# ── Unified leg normalizer ────────────────────────────────────────────────────

def _to_parlay_leg(p: dict) -> dict:
    """Normalize a prop dict or alt-line dict into a unified parlay leg format.

    Works with both regular props (which have model_over_odds / model_under_odds)
    and alt-line entries (which may carry a pre-computed model_odds key).
    """
    direction = p.get("direction", "Over")
    # For regular props, pick the odds side matching the direction
    if "model_odds" in p:
        model_odds = p["model_odds"]
    elif direction.lower() == "over":
        model_odds = p.get("model_over_odds", -110)
    else:
        # Under props: model_prob IS the under probability, so model_over_odds
        # is already _prob_to_american(model_prob) — the correct line to use.
        model_odds = p.get("model_over_odds", p.get("model_under_odds", -110))

    return {
        "player":     p["player"],
        "stat":       p.get("stat") or p.get("stat_label", "?"),
        "direction":  direction,
        "line":       float(p.get("line") or p.get("threshold") or 0),
        "win_prob":   round(p.get("model_prob", p.get("win_prob", 0.65)) * 100, 1),
        "model_odds": model_odds,
        "is_lock":    p.get("is_lock", False),
        "team":       p.get("team", ""),
        "opponent":   p.get("opponent", ""),
        "hit_rate":   p.get("hit_rate"),
        "ev":         p.get("ev"),
        "game":       p.get("game_matchup", p.get("game", "")),
    }


# ── Best Bets (10 single high-confidence picks) ───────────────────────────────

def build_best_bets(props: list[dict], n: int = 10) -> list[dict]:
    """Select top n individual props with highest model confidence.

    Returns raw prop dicts (not wrapped in parlay format) sorted by:
        1. Locks first (hit_rate >= 80%)
        2. model_prob descending
        3. EV descending

    Deduplicates by player — max 1 pick per player.
    Primary threshold: model_prob >= 0.63; relaxes to 0.58 if fewer than n found.
    """
    def _eligible(p: dict, min_prob: float) -> bool:
        return (
            p.get("model_prob", 0) >= min_prob
            and not p.get("blowout_risk", False)
            and not p.get("is_combo", False)
        )

    def _sort_key(p: dict) -> tuple:
        return (not p.get("is_lock", False), -p.get("model_prob", 0), -p.get("ev", 0))

    # Primary pass: 0.63+
    eligible = sorted([p for p in props if _eligible(p, 0.63)], key=_sort_key)
    seen_players: set[str] = set()
    best_bets: list[dict] = []

    for p in eligible:
        if p["player"] not in seen_players:
            seen_players.add(p["player"])
            best_bets.append(p)
        if len(best_bets) == n:
            break

    # Fallback pass: 0.58+ if still short
    if len(best_bets) < n:
        fallback = sorted(
            [p for p in props if _eligible(p, 0.58) and p["player"] not in seen_players],
            key=_sort_key,
        )
        for p in fallback:
            seen_players.add(p["player"])
            best_bets.append(p)
            if len(best_bets) == n:
                break

    return best_bets[:n]


# ── 2-Leg parlays ─────────────────────────────────────────────────────────────

def build_two_leg_parlays(
    props: list[dict],
    tracker: _DiversityTracker,
    n: int = 10,
) -> list[dict]:
    """Build n two-legged parlays from the highest-quality prop pool.

    Eligibility: model_prob >= 0.60, no blowout_risk.
    Minimum combined win probability: 32% (e.g. two 57% legs = 32.5%).
    Priority ordering: locks first, then by model_prob descending.
    """
    pool = sorted(
        [p for p in props
         if p.get("model_prob", 0) >= 0.60
         and not p.get("blowout_risk", False)],
        key=lambda p: (-int(p.get("is_lock", False)), -p.get("model_prob", 0)),
    )

    parlays: list[dict] = []
    used_pairs: set[frozenset] = set()

    for i, leg_a in enumerate(pool):
        if len(parlays) == n:
            break
        player_a = leg_a["player"]
        stat_a   = leg_a.get("stat_type") or leg_a.get("stat", "")
        if not tracker.can_use(player_a, stat_a, leg_a.get("direction", "")):
            continue

        for leg_b in pool[i + 1:]:
            if leg_b["player"] == player_a:
                continue

            player_b = leg_b["player"]
            stat_b   = leg_b.get("stat_type") or leg_b.get("stat", "")
            if not tracker.can_use(player_b, stat_b, leg_b.get("direction", "")):
                continue

            pair = frozenset([(player_a, stat_a), (player_b, stat_b)])
            if pair in used_pairs:
                continue

            legs = [_to_parlay_leg(leg_a), _to_parlay_leg(leg_b)]
            odds = parlay_odds(legs)

            if odds["win_prob"] < 32.0:
                continue

            used_pairs.add(pair)
            tracker.mark_used(player_a, stat_a, leg_a.get("direction", ""))
            tracker.mark_used(player_b, stat_b, leg_b.get("direction", ""))

            parlays.append({
                "label": f"2-Leg Parlay #{len(parlays) + 1}",
                "type":  "two_leg",
                "legs":  legs,
                "odds":  odds,
            })
            break

    return parlays


# ── 3-Leg unified parlays ─────────────────────────────────────────────────────

def build_three_leg_parlays(
    props: list[dict],
    alt_lines: list[dict],
    tracker: _DiversityTracker,
    n: int = 10,
) -> list[dict]:
    """Build n unified 3-leg parlays from props + alt lines (direction-agnostic).

    Unified pool is sorted: locks first → model_prob → EV.
    Greedy algorithm: for each anchor, picks the best 2 player-diverse companions.
    Minimum combined win probability: 18% (e.g. three 56% legs ≈ 17.6%).
    """
    # Build unified pool from regular props
    pool: list[dict] = []
    for p in props:
        if p.get("model_prob", 0) >= 0.56 and not p.get("blowout_risk", False):
            direction = p.get("direction", "Over")
            model_odds = (
                p.get("model_over_odds", -110)
                if direction.lower() == "over"
                else p.get("model_over_odds", p.get("model_under_odds", -110))
            )
            pool.append({
                **p,
                "source":     "prop",
                "model_odds": model_odds,
            })

    # Add alt lines converted to the common format
    for a in alt_lines:
        window = a.get("window", 5)
        prob   = _streak_prob(window)
        pool.append({
            **a,
            "source":     "alt",
            "direction":  "Over",
            "line":       float(a.get("threshold", 0)),
            "model_prob": prob,
            "ev":         prob - 0.52,
            "is_lock":    True,
            "model_odds": _prob_to_american(prob),
            "blowout_risk": False,
        })

    # Sort: locks first → model_prob → EV
    pool.sort(key=lambda p: (
        -int(p.get("is_lock", False)),
        -p.get("model_prob", 0),
        -p.get("ev", 0),
    ))

    parlays: list[dict] = []
    used_triplets: set[frozenset] = set()

    for anchor in pool:
        if len(parlays) == n:
            break

        player_a = anchor["player"]
        stat_a   = anchor.get("stat_type") or anchor.get("stat", "")
        if not tracker.can_use(player_a, stat_a, anchor.get("direction", "")):
            continue

        # Find best 2 player-diverse companions
        companions: list[dict] = []
        for candidate in pool:
            if candidate["player"] == player_a:
                continue
            if any(c["player"] == candidate["player"] for c in companions):
                continue
            c_stat = candidate.get("stat_type") or candidate.get("stat", "")
            if not tracker.can_use(candidate["player"], c_stat, candidate.get("direction", "")):
                continue
            companions.append(candidate)
            if len(companions) == 2:
                break

        if len(companions) < 2:
            continue

        triplet = frozenset([
            (player_a, stat_a),
            (companions[0]["player"], companions[0].get("stat_type") or companions[0].get("stat", "")),
            (companions[1]["player"], companions[1].get("stat_type") or companions[1].get("stat", "")),
        ])
        if triplet in used_triplets:
            continue

        legs = [_to_parlay_leg(anchor), _to_parlay_leg(companions[0]), _to_parlay_leg(companions[1])]
        odds = parlay_odds(legs)

        if odds["win_prob"] < 18.0:
            continue

        used_triplets.add(triplet)
        tracker.mark_used(player_a, stat_a, anchor.get("direction", ""))
        for c in companions:
            c_stat = c.get("stat_type") or c.get("stat", "")
            tracker.mark_used(c["player"], c_stat, c.get("direction", ""))

        parlays.append({
            "label": f"3-Leg Parlay #{len(parlays) + 1}",
            "type":  "three_leg",
            "legs":  legs,
            "odds":  odds,
        })

    return parlays


# ── Safe Over parlays (reduced-average lines) ────────────────────────────────

# Minimum absolute reduction per stat type
_SAFE_REDUCE_MIN: dict[str, float] = {
    "PTS": 4.0,
    "AST": 1.5,
    "REB": 1.5,
    "FG3M": 0.5,
}
_SAFE_REDUCE_PCT = 0.20   # 20% below L5 avg
_SAFE_PROB       = 0.82   # estimated hit probability at reduced line


def build_reduced_avg_parlays(
    props: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 10,
    n_legs: int = 3,
) -> list[dict]:
    """Build 10x 3-leg 'safe over' parlays using lines ~20% below player's L5 avg.

    Concept: if a player averages 25 pts in last 5-10 games, take them for
    Over 20.  If they avg 6 ast → Over 4.  If they avg 10 reb → Over 8.
    Line = floor((l5_avg - reduction) × 2) / 2  (rounded down to nearest 0.5).

    Only single stats (no combo), no BLK/STL (those live in defense parlays).
    Reduced line must be strictly below the book line to ensure value.
    """
    pool: list[dict] = []
    seen: set[tuple] = set()   # deduplicate (player, stat) in pool

    for prop in props:
        stat = prop.get("stat_type") or prop.get("stat", "")

        # Skip combo stats and defense stats
        if "+" in stat or stat in ("BLK", "STL"):
            continue

        l5_avg = float(prop.get("l5_avg") or prop.get("avg") or 0.0)
        if l5_avg < 4.0:
            continue

        player = prop["player"]
        key = (player.lower(), stat.upper())
        if key in seen:
            continue
        seen.add(key)

        min_reduce = _SAFE_REDUCE_MIN.get(stat, 1.0)
        reduction  = max(l5_avg * _SAFE_REDUCE_PCT, min_reduce)
        raw_line   = l5_avg - reduction
        red_line   = math.floor(raw_line * 2) / 2   # round down to nearest 0.5

        if red_line < 0.5:
            continue

        book_line = float(prop.get("line") or 0.0)
        # Reduced line must be below the book's current line
        if book_line > 0 and red_line >= book_line:
            continue

        model_odds = _prob_to_american(_SAFE_PROB)

        pool.append({
            **prop,
            # Override the line so parlay display shows the reduced line
            "line":       red_line,
            "direction":  "Over",
            "model_odds": model_odds,
            "win_prob":   round(_SAFE_PROB * 100, 1),
            "model_prob": _SAFE_PROB,
            "is_lock":    False,
            "ev":         round(_SAFE_PROB - 0.52, 3),
            # Keep original book line for reference in display
            "book_line":  book_line,
            "l5_avg_used": round(l5_avg, 1),
        })

    # Sort: highest L5 average first (most dominant players lead)
    pool.sort(key=lambda x: -x.get("l5_avg_used", 0))

    # Re-use generic builder — each leg uses pre-computed model_odds
    parlays: list[dict] = []

    for parlay_idx in range(n_parlays):
        legs: list[dict] = []
        used_this: set[tuple] = set()

        for p in pool:
            if len(legs) >= n_legs:
                break

            player = p["player"]
            stat   = p.get("stat_type") or p.get("stat", "")
            key    = (player.lower(), stat.upper())

            if key in used_this:
                continue
            if not tracker.can_use(player, stat, "over"):
                continue

            legs.append({
                "player":     player,
                "team":       p.get("team", ""),
                "stat":       stat,
                "stat_label": stat,
                "direction":  "Over",
                "line":       p["line"],       # reduced line
                "book_line":  p["book_line"],  # original book line
                "l5_avg":     p["l5_avg_used"],
                "win_prob":   p["win_prob"],
                "model_odds": p["model_odds"],
                "hit_rate":   p.get("hit_rate"),
                "ev":         p["ev"],
                "label":      (
                    f"{player} Over {p['line']} {stat}  "
                    f"(avg {p['l5_avg_used']})"
                ),
                "game":       p.get("game_matchup", ""),
            })
            used_this.add(key)

        if len(legs) >= n_legs:
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], "over")
            parlays.append({
                "label": f"Safe Over Parlay #{parlay_idx + 1}  (Reduced Lines)",
                "type":  "reduced",
                "legs":  legs,
                "odds":  parlay_odds(legs),
            })

    return parlays


# ── Master builder ────────────────────────────────────────────────────────────

def build_all_parlays(
    props: list[dict],
    alt_lines: list[dict],
    game_predictions: list[dict],
) -> dict:
    """Build all parlay groups in one call.

    Args:
        props:            Main page props (with model_over_odds / model_under_odds).
        alt_lines:        100% Alt Line streaks (with stat, threshold, window).
        game_predictions: Game-level predictions (with winner_pick, winner_confidence).

    Returns dict:
        {
            "best_bets":   [prop dict, ...],     # up to 10 single picks
            "two_leg":     [parlay dict, ...],   # up to 10 two-leg parlays
            "three_leg":   [parlay dict, ...],   # up to 10 three-leg parlays
            "ml":          parlay dict | None,
            "alt":         [parlay dict, ...],   # up to 2
            "over":        [parlay dict, ...],   # up to 5
            "under":       [parlay dict, ...],   # up to 3
            "defense":     [parlay dict, ...],   # up to 3
            "total_count": int,
        }
    """
    tracker = _DiversityTracker(max_uses=2)

    # Build order: prop parlays first (best picks), then alt/defense (wider pool)
    ml_parlay        = build_ml_parlay(game_predictions, tracker)
    over_parlays     = build_over_parlays(props,      tracker, n_parlays=5,  n_legs=3)
    under_parlays    = build_under_parlays(props,     tracker, n_parlays=3,  n_legs=3)
    reduced_parlays  = build_reduced_avg_parlays(props, tracker, n_parlays=10, n_legs=3)
    alt_parlays      = build_alt_parlays(alt_lines,   tracker, n_parlays=2,  n_legs=10)
    defense_parlays  = build_defense_parlays(alt_lines, tracker, n_parlays=3, n_legs=3)

    total = (
        (1 if ml_parlay else 0)
        + len(over_parlays)
        + len(under_parlays)
        + len(reduced_parlays)
        + len(alt_parlays)
        + len(defense_parlays)
    )

    return {
        "ml":          ml_parlay,
        "alt":         alt_parlays,
        "over":        over_parlays,
        "under":       under_parlays,
        "reduced":     reduced_parlays,
        "defense":     defense_parlays,
        "total_count": total,
    }
