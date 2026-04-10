# Implementation Plan: Props Balance + Parlay Builder Overhaul

## Summary of Changes

Six targeted changes across three files + one new file:

1. **Fix pick count** — 25 → 60 hard cap
2. **Fix Alt line thresholds** — raise minimums to betting-viable levels + add BLK/STL
3. **Add model-generated odds** — convert model probability → American odds with vig
4. **New `utils/parlay_builder.py`** — all parlay logic isolated here
5. **Cache parlays** — add to `refresh_props_cache()` output
6. **New "Parlays" dashboard tab** — dedicated view with all parlay sections

---

## Problem Analysis

### Pick Count (25 is too low)
- `_HARD_CAP = 25` at `props_cache.py:629`
- On a full 10-game NBA night there are ~150 qualified players → 25 caps too aggressively
- Target: **60 props** with `_MAX_PER_PLAYER = 2` still in place (prevents any single hot player flooding)

### Alt Lines Not Bet-Viable
- Current `_ALT_MIN_THRESH = {"PTS": 5, "AST": 2, "REB": 3, "FG3M": 1}` at `props_cache.py:147`
- Problem: The threshold shown is the **minimum hit in the streak** — e.g., Kon Knueppel hit 7+ in 20 straight games. But books only offer him at 10.5+ because he's a starter averaging ~14 PPG
- Fix: Raise global floors to what sportsbooks actually offer, and add a context check

### No Odds Generation
- Props show model probability but no estimated odds → can't calculate parlay payouts
- Need: `model_over_odds` field on each prop using the same vig math books use

---

## Implementation Steps

---

### STEP 1: Fix Pick Count
**File:** `utils/props_cache.py`

**Change 1a** — Raise hard cap (line 629):
```python
# BEFORE
_HARD_CAP = 25

# AFTER
_HARD_CAP = 60
```

**Change 1b** — Slightly lower model prob floor to allow more through quality gate (line 626):
```python
# BEFORE
_MIN_MODEL_PROB = 0.58

# AFTER
_MIN_MODEL_PROB = 0.56   # Opens ~10-15 more picks without sacrificing quality
```

**Result:** ~50-65 props on a full game night, ~25-35 on a light night.

---

### STEP 2: Fix Alt Line Thresholds + Add BLK/STL
**File:** `utils/props_cache.py`

**Change 2a** — Raise thresholds to bet-viable minimums (line 147):
```python
# BEFORE
_ALT_MIN_THRESH = {"PTS": 5, "AST": 2, "REB": 3, "FG3M": 1}
_ALT_STAT_LABELS = {"PTS": "POINTS", "AST": "ASSISTS", "REB": "REBOUNDS", "FG3M": "MADE THREES"}

# AFTER
_ALT_MIN_THRESH = {
    "PTS":  10,   # Books don't offer lines below 10.5 for anyone meaningful
    "AST":  3,    # Books rarely offer under 2.5 assists
    "REB":  4,    # Books rarely offer under 3.5 rebounds
    "FG3M": 2,    # Books offer 1.5+, floor at 2 makes streak meaningful
    "BLK":  1,    # Needed for defense parlays (1+ block in every game)
    "STL":  1,    # Needed for defense parlays (1+ steal in every game)
}
_ALT_STAT_LABELS = {
    "PTS": "POINTS", "AST": "ASSISTS", "REB": "REBOUNDS",
    "FG3M": "MADE THREES", "BLK": "BLOCKS", "STL": "STEALS",
}
```

**Change 2b** — Add context-aware viability filter inside `_compute_alt_lines()` (after line 1001, before `alt_lines.append(...)`):
```python
# Context check: threshold must be >= 55% of player's season average
# Prevents showing "10+ PTS for a 10.5 PPG player" — technically true but the
# line IS basically their average and books won't have a lower line available
season_avg = float(stat_series.mean())
if season_avg > 0 and (best_thresh / season_avg) < 0.55:
    continue  # Too close to their average, not a meaningful streak
```

**Why 55%:** If a player averages 20 PTS and hits 10+ in every game, the streak threshold (10) is exactly half their average. That's strong — a clear floor. But if they average 12 and hit 10+ every game, the threshold is 83% of their average, which is fine. The 55% floor ensures we're showing streaks that are at least somewhat below the player's norm, creating real betting value.

---

### STEP 3: Add Model-Generated Odds
**File:** `utils/props_cache.py`

**Change 3a** — Add odds conversion function near top of file (after line 150):
```python
def _prob_to_american(prob: float, vig: float = 0.0476) -> int:
    """Convert true probability → American odds with sportsbook vig applied.

    Books apply vig by inflating implied probability so both sides sum to > 100%.
    Standard -110/-110 line creates 52.38% + 52.38% = 104.76% implied, i.e. 4.76% vig.

    Args:
        prob: True win probability (0-1)
        vig:  Vig rate to apply (default 4.76% = standard -110 market)

    Returns: American odds integer (e.g. -150, +130)
    """
    # Clamp to avoid division by zero
    prob = max(0.01, min(0.99, prob))
    # Apply vig — inflate the probability (makes odds worse for bettor, better for book)
    viggged = prob * (1 + vig)
    viggged = min(viggged, 0.99)
    if viggged >= 0.5:
        return int(-100 * viggged / (1 - viggged))
    else:
        return int(100 * (1 - viggged) / viggged)
```

**Change 3b** — Apply to each prop in `_compute_main_page_props()`, inside the props loop where we build the prop dict (around line 418), add:
```python
"model_over_odds":  _prob_to_american(model_prob),
"model_under_odds": _prob_to_american(1 - model_prob),
```

**Examples:**
- 70% model prob → viggged 73.3% → **-275** (solid favorite)
- 60% model prob → viggged 62.9% → **-170** (moderate favorite)
- 55% model prob → viggged 57.6% → **-136** (slight edge)

---

### STEP 4: New `utils/parlay_builder.py`
**New file:** `utils/parlay_builder.py`

Full module structure:

```python
"""
parlay_builder.py
Constructs recommended parlays from the model's best props and game predictions.

Parlays produced:
    - 1x  3-leg Moneyline parlay (from game predictions)
    - 2x 10-leg 100% Alt Line parlays
    - 5x  3-leg Props OVER parlays
    - 3x  3-leg Props UNDER parlays
    - 3x  3-leg Defense (BLK/STL only) parlays

Diversity rule: same (player, stat, direction) combo may appear in at most 2 parlays.
"""

import math
from collections import defaultdict
from typing import Optional

# ── Odds math ─────────────────────────────────────────────────────────────────

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds."""
    if american > 0:
        return american / 100 + 1.0
    else:
        return 100 / abs(american) + 1.0

def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds back to American."""
    if decimal >= 2.0:
        return int((decimal - 1) * 100)
    else:
        return int(-100 / (decimal - 1))

def parlay_odds(legs: list[dict]) -> dict:
    """
    Calculate combined parlay odds from a list of legs.

    Each leg must have 'model_odds' key (American int).
    Returns: {
        'decimal': float,
        'american': int,
        'win_prob': float,     # combined probability (product of individual probs)
        'payout_100': float,   # payout on $100 bet
    }
    """
    decimal = 1.0
    win_prob = 1.0
    for leg in legs:
        odds = leg.get("model_odds", -110)
        dec = american_to_decimal(odds)
        decimal *= dec
        # True prob from american odds (remove vig approximation)
        if odds < 0:
            win_prob *= abs(odds) / (abs(odds) + 100)
        else:
            win_prob *= 100 / (odds + 100)
    return {
        "decimal": round(decimal, 2),
        "american": decimal_to_american(decimal),
        "win_prob": round(win_prob * 100, 1),
        "payout_100": round((decimal - 1) * 100, 2),
    }

# ── Diversity tracker ─────────────────────────────────────────────────────────

class _DiversityTracker:
    """Enforces: same (player, stat, direction) appears in at most 2 parlays."""

    def __init__(self, max_uses: int = 2):
        self._uses: dict = defaultdict(int)
        self._max = max_uses

    def can_use(self, player: str, stat: str, direction: str) -> bool:
        key = (player.lower(), stat.upper(), direction.lower())
        return self._uses[key] < self._max

    def mark_used(self, player: str, stat: str, direction: str) -> None:
        key = (player.lower(), stat.upper(), direction.lower())
        self._uses[key] += 1

# ── Parlay builders ───────────────────────────────────────────────────────────

def build_ml_parlay(game_predictions: list[dict], tracker: "_DiversityTracker") -> Optional[dict]:
    """
    Build a 3-leg Moneyline parlay from today's game predictions.

    Args:
        game_predictions: List of game dicts with keys:
            home, away, winner_pick, winner_confidence,
            home_win_prob (0-100)
        tracker: Shared diversity tracker

    Returns: Parlay dict with 'legs', 'odds', 'label' or None if < 3 picks available.
    """
    # Confidence → win probability mapping (from app.py line 1359)
    CONF_TO_PROB = {"HIGH": 0.78, "MEDIUM": 0.62, "LOW": 0.52}

    # Sort by confidence descending
    picks = []
    for g in game_predictions:
        conf = g.get("winner_confidence", "LOW")
        if conf == "LOW":
            continue  # Skip LOW confidence ML picks for parlays
        prob = CONF_TO_PROB.get(conf, 0.52)
        winner = g.get("winner_pick")
        if not winner:
            continue

        # Adjust probability if away team is the pick
        home = g.get("home", "")
        if winner != home:
            prob = 1 - prob  # Flip for away team perspective
            prob = max(0.52, prob)

        model_odds = _prob_to_american_local(prob)

        picks.append({
            "player": f"{g.get('away')} @ {g.get('home')}",
            "stat": "ML",
            "direction": "ML",
            "pick": winner,
            "confidence": conf,
            "win_prob": round(prob * 100, 1),
            "model_odds": model_odds,
            "label": f"{winner} ML",
            "game": f"{g.get('away')} @ {g.get('home')}",
        })

    # Sort by probability descending, take top 3
    picks.sort(key=lambda x: -x["win_prob"])
    legs = picks[:3]

    if len(legs) < 3:
        return None

    # Register usage (ML picks don't use diversity tracker — they're game-level)
    odds = parlay_odds(legs)
    return {
        "label": "3-Leg Moneyline Parlay",
        "type": "ml",
        "legs": legs,
        "odds": odds,
    }


def build_alt_parlays(
    alt_lines: list[dict],
    tracker: "_DiversityTracker",
    n_parlays: int = 2,
    n_legs: int = 10,
) -> list[dict]:
    """
    Build n_parlays 10-leg parlays from 100% Alt Lines.

    Alt lines don't have individual model_odds — we use the streak length
    as a proxy: longer streak = higher implied probability:
      5-game streak  → 70% (needs 5/5) = 0.7^5 independent ≈ 16.8% parlay prob if independent
      But each game isn't independent — we use the streak as confidence signal:
      window 5-7   → 72% per leg
      window 8-12  → 78% per leg
      window 13-17 → 84% per leg
      window 18-20 → 88% per leg
    """
    WINDOW_TO_PROB = {
        5: 0.72, 6: 0.72, 7: 0.72,
        8: 0.78, 10: 0.78, 12: 0.78,
        15: 0.84, 17: 0.84,
        18: 0.88, 20: 0.88,
    }

    parlays = []
    used_in_this_batch: set = set()

    for parlay_idx in range(n_parlays):
        legs = []

        for alt in alt_lines:
            if len(legs) >= n_legs:
                break

            player = alt["player"]
            stat   = alt["stat"]
            key    = (player, stat, parlay_idx)

            # Diversity: same (player, stat) can't appear in same parlay twice (obvious)
            if (player, stat) in {(l["player"], l["stat"]) for l in legs}:
                continue

            # Diversity: same (player, stat) in at most 2 parlays total
            if not tracker.can_use(player, stat, "alt"):
                continue

            window = alt["window"]
            # Find closest window key
            closest = min(WINDOW_TO_PROB.keys(), key=lambda w: abs(w - window))
            prob = WINDOW_TO_PROB[closest]
            odds = _prob_to_american_local(prob)

            leg = {
                "player":    player,
                "team":      alt["team"],
                "opponent":  alt["opponent"],
                "stat":      stat,
                "stat_label": alt["stat_label"],
                "direction": "Over",
                "threshold": alt["threshold"],
                "window":    window,
                "trend":     alt["trend"],
                "win_prob":  round(prob * 100, 1),
                "model_odds": odds,
                "label":     f"{player} {alt['threshold']}+ {alt['stat_label']}",
                "game":      alt.get("game_matchup", ""),
            }
            legs.append(leg)

        if len(legs) >= 8:  # Require at least 8 legs for an alt parlay
            # Register usage
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], "alt")

            odds = parlay_odds(legs)
            parlays.append({
                "label": f"10-Leg Alt Lines Parlay #{parlay_idx + 1}",
                "type":  "alt",
                "legs":  legs,
                "odds":  odds,
            })

    return parlays


def build_over_parlays(
    props: list[dict],
    tracker: "_DiversityTracker",
    n_parlays: int = 5,
    n_legs: int = 3,
) -> list[dict]:
    """Build 5 three-leg OVER parlays from best props."""
    # Filter to Overs only, sorted by model_prob descending
    overs = [
        p for p in props
        if p.get("direction", "").lower() == "over"
        and p.get("model_prob", 0) >= 0.56
        and p.get("model_over_odds") is not None
    ]
    overs.sort(key=lambda x: -(x.get("model_prob") or 0))

    return _build_prop_parlays(
        overs, tracker, direction="over",
        n_parlays=n_parlays, n_legs=n_legs,
        parlay_type="over", label_prefix="Props OVER Parlay"
    )


def build_under_parlays(
    props: list[dict],
    tracker: "_DiversityTracker",
    n_parlays: int = 3,
    n_legs: int = 3,
) -> list[dict]:
    """Build 3 three-leg UNDER parlays from best props."""
    unders = [
        p for p in props
        if p.get("direction", "").lower() == "under"
        and p.get("model_prob", 0) >= 0.56
        and p.get("model_under_odds") is not None
    ]
    unders.sort(key=lambda x: -(x.get("model_prob") or 0))

    return _build_prop_parlays(
        unders, tracker, direction="under",
        n_parlays=n_parlays, n_legs=n_legs,
        parlay_type="under", label_prefix="Props UNDER Parlay"
    )


def build_defense_parlays(
    alt_lines: list[dict],
    tracker: "_DiversityTracker",
    n_parlays: int = 3,
    n_legs: int = 3,
) -> list[dict]:
    """
    Build 3 three-leg DEFENSE parlays using only BLK and STL alt lines.
    Uses the same streak-to-probability mapping as alt parlays.
    """
    defense_alts = [a for a in alt_lines if a["stat"] in ("BLK", "STL")]

    WINDOW_TO_PROB = {
        5: 0.72, 6: 0.72, 7: 0.72,
        8: 0.78, 10: 0.78, 12: 0.78,
        15: 0.84, 17: 0.84, 18: 0.88, 20: 0.88,
    }

    parlays = []
    for parlay_idx in range(n_parlays):
        legs = []
        for alt in defense_alts:
            if len(legs) >= n_legs:
                break
            player = alt["player"]
            stat   = alt["stat"]
            if (player, stat) in {(l["player"], l["stat"]) for l in legs}:
                continue
            if not tracker.can_use(player, stat, "def"):
                continue

            closest = min(WINDOW_TO_PROB.keys(), key=lambda w: abs(w - alt["window"]))
            prob    = WINDOW_TO_PROB[closest]
            odds    = _prob_to_american_local(prob)

            legs.append({
                "player":     player,
                "team":       alt["team"],
                "stat":       stat,
                "stat_label": alt["stat_label"],
                "direction":  "Over",
                "threshold":  alt["threshold"],
                "window":     alt["window"],
                "trend":      alt["trend"],
                "win_prob":   round(prob * 100, 1),
                "model_odds": odds,
                "label":      f"{player} {alt['threshold']}+ {alt['stat_label']}",
                "game":       alt.get("game_matchup", ""),
            })

        if len(legs) == n_legs:
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], "def")
            odds = parlay_odds(legs)
            parlays.append({
                "label": f"Defense Parlay #{parlay_idx + 1} (BLK/STL)",
                "type":  "defense",
                "legs":  legs,
                "odds":  odds,
            })

    return parlays


def _build_prop_parlays(
    sorted_props: list[dict],
    tracker: "_DiversityTracker",
    direction: str,
    n_parlays: int,
    n_legs: int,
    parlay_type: str,
    label_prefix: str,
) -> list[dict]:
    """Generic prop parlay builder used by build_over/under_parlays."""
    parlays = []
    odds_key = "model_over_odds" if direction == "over" else "model_under_odds"

    for parlay_idx in range(n_parlays):
        legs = []
        for prop in sorted_props:
            if len(legs) >= n_legs:
                break
            player = prop["player"]
            stat   = prop.get("stat_type", prop.get("stat", ""))
            if not tracker.can_use(player, stat, direction):
                continue
            # No duplicate player-stat within same parlay
            if (player, stat) in {(l["player"], l["stat"]) for l in legs}:
                continue

            model_odds = prop.get(odds_key)
            if model_odds is None:
                continue

            prob = prop.get("model_prob", 0.56)
            legs.append({
                "player":     player,
                "team":       prop.get("team", ""),
                "stat":       stat,
                "direction":  direction.capitalize(),
                "line":       prop.get("line"),
                "win_prob":   round(prob * 100, 1),
                "model_odds": model_odds,
                "hit_rate":   prop.get("hit_rate"),
                "ev":         prop.get("ev"),
                "label":      f"{player} {direction.capitalize()} {prop.get('line')} {stat}",
                "game":       prop.get("game_matchup", ""),
            })

        if len(legs) == n_legs:
            for leg in legs:
                tracker.mark_used(leg["player"], leg["stat"], direction)
            odds = parlay_odds(legs)
            parlays.append({
                "label": f"{label_prefix} #{parlay_idx + 1}",
                "type":  parlay_type,
                "legs":  legs,
                "odds":  odds,
            })

    return parlays


def _prob_to_american_local(prob: float, vig: float = 0.0476) -> int:
    """Local copy to avoid circular import."""
    prob = max(0.01, min(0.99, prob))
    viggged = min(prob * (1 + vig), 0.99)
    if viggged >= 0.5:
        return int(-100 * viggged / (1 - viggged))
    return int(100 * (1 - viggged) / viggged)


def build_all_parlays(
    props: list[dict],
    alt_lines: list[dict],
    game_predictions: list[dict],
) -> dict:
    """
    Master builder — returns all parlay groups.

    Returns:
    {
        "ml":      parlay dict or None,
        "alt":     [parlay dict, ...],
        "over":    [parlay dict, ...],
        "under":   [parlay dict, ...],
        "defense": [parlay dict, ...],
        "total_count": int
    }
    """
    tracker = _DiversityTracker(max_uses=2)

    ml_parlay      = build_ml_parlay(game_predictions, tracker)
    alt_parlays    = build_alt_parlays(alt_lines,  tracker, n_parlays=2,  n_legs=10)
    over_parlays   = build_over_parlays(props,     tracker, n_parlays=5,  n_legs=3)
    under_parlays  = build_under_parlays(props,    tracker, n_parlays=3,  n_legs=3)
    defense_parlays = build_defense_parlays(alt_lines, tracker, n_parlays=3, n_legs=3)

    total = (
        (1 if ml_parlay else 0)
        + len(alt_parlays)
        + len(over_parlays)
        + len(under_parlays)
        + len(defense_parlays)
    )

    return {
        "ml":          ml_parlay,
        "alt":         alt_parlays,
        "over":        over_parlays,
        "under":       under_parlays,
        "defense":     defense_parlays,
        "total_count": total,
    }
```

---

### STEP 5: Cache Parlays in `refresh_props_cache()`
**File:** `utils/props_cache.py`

After the existing `_compute_alt_lines()` call in `refresh_props_cache()` (around line 1086), add:

```python
from utils.parlay_builder import build_all_parlays

# Flatten game predictions from game_info for ML parlay
game_preds = []
for away_team, home_team in game_info.get("game_pairs", []):
    # Pull pre-computed prediction if available
    game_preds.append({
        "home": home_team,
        "away": away_team,
        "winner_pick": game_info.get("winner_picks", {}).get(f"{away_team}@{home_team}"),
        "winner_confidence": game_info.get("winner_confidences", {}).get(f"{away_team}@{home_team}", "LOW"),
    })

parlays = build_all_parlays(
    props=main_props,
    alt_lines=alt_lines,
    game_predictions=game_preds,
)

_props_cache["parlays"] = parlays
```

Also expose a getter:
```python
def get_parlays_cache() -> dict:
    """Return the parlays dict from the last cache refresh."""
    return _props_cache.get("parlays", {})
```

---

### STEP 6: New Dashboard "Parlays" Tab
**File:** `dashboard/app.py`

#### 6a — Add 4th tab to view switcher (around line 1762):
```python
html.Div("Props",          id="props-view-tab-props",    n_clicks=0, className="view-tab active"),
html.Div("100% Alt Lines", id="props-view-tab-alt",      n_clicks=0, className="view-tab"),
html.Div("Record ✓",       id="props-view-tab-record",   n_clicks=0, className="view-tab"),
html.Div("Parlays 🎯",     id="props-view-tab-parlays",  n_clicks=0, className="view-tab"),  # NEW
```

#### 6b — Update `update_props_view()` callback (around line 2355):
- Add `Input("props-view-tab-parlays", "n_clicks")`
- Return `"parlays"` when that tab is triggered
- Return 4 className values + filter panel visibility

#### 6c — Update `update_props_list()` callback:
After the existing Record/Alt/Props branches, add:
```python
if view == "parlays":
    from utils.props_cache import get_parlays_cache
    parlays = get_parlays_cache()
    return _create_parlays_section(parlays)
```

#### 6d — New `_create_parlays_section(parlays)` function:
Structure:
```
_create_parlays_section(parlays: dict) -> html.Div:
    Sections in order (each on its own visual card):
    1. "Moneyline Parlay" — 3-leg ML
    2. "10-Leg Alt Lines Parlay #1"
    3. "10-Leg Alt Lines Parlay #2"
    4. "Props OVER Parlay #1" through "#5"
    5. "Props UNDER Parlay #1" through "#3"
    6. "Defense Parlay #1" through "#3" (BLK/STL)

    Each card contains:
    - Header: parlay label + combined odds badge + win probability
    - Leg list: player, prop, line, direction, individual odds
    - Footer: "Est. Payout on $100: $X.XX"
```

#### Parlay Card Visual Design:
```
┌────────────────────────────────────────────────────────┐
│  🎯 Props OVER Parlay #1          +524   Win: 34.2%  │
├────────────────────────────────────────────────────────┤
│  LaMelo Ball    Over 26.5 PTS    -162   72% hit rate  │
│  Jayson Tatum   Over 8.5 REB     -148   68% hit rate  │
│  Tyrese Haliburton Over 9.5 AST  -155   70% hit rate  │
├────────────────────────────────────────────────────────┤
│  Est. Payout on $100: $524                             │
└────────────────────────────────────────────────────────┘
```

---

## Key Files Summary

| File | Operation | Lines Affected |
|------|-----------|----------------|
| `utils/props_cache.py` | Modify | 147, 150-165 (new fn), 626, 629, 1001-1002, 1086-1095 |
| `utils/parlay_builder.py` | **Create** | Full new file |
| `dashboard/app.py` | Modify | 1762-1765, 2355-2367, 2562-2600 + new `_create_parlays_section()` |

---

## Validation Checklist

After implementation:
- [ ] `_HARD_CAP = 60` and seeing 50-65 picks on full game nights
- [ ] Alt lines: No threshold below 10 for PTS, 3 for AST, 4 for REB, 2 for FG3M
- [ ] Alt lines: BLK and STL streaks appear in the Alt Lines tab
- [ ] Alt lines: The 55% context check filters out trivial streaks
- [ ] Each prop shows `model_over_odds` / `model_under_odds` in cache
- [ ] Parlay tab visible in UI
- [ ] 3-leg ML parlay present (or None if < 3 HIGH/MEDIUM picks)
- [ ] 2x 10-leg Alt parlays (or fewer if not enough Alt lines)
- [ ] 5x 3-leg Over parlays
- [ ] 3x 3-leg Under parlays
- [ ] 3x 3-leg Defense (BLK/STL) parlays
- [ ] Diversity: Same player-stat-direction combo in max 2 parlays
- [ ] Combined parlay odds calculate correctly

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (single-model plan)
- GEMINI_SESSION: N/A (single-model plan)
