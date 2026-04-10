# Parlay System Overhaul v2

**Feature:** Clean display, fix overs, 29 distinct parlays, max-2-per-player globally
**Status:** Planning
**Date:** 2026-04-08

---

## Exact Parlay Count

| Section | Count | Type | Description |
|---------|-------|------|-------------|
| Main Overs | 10 | 3-leg Props Over | Best over props by model probability |
| Moneyline | 3 | 3-leg ML | Top game winner picks |
| Total Points | 3 | 3-leg Game Totals | Over/Under game total picks |
| Spread | 3 | 3-leg Spread/ATS | Home/Away covers |
| Alt Overs | 5 | 3-leg Props Over | Secondary over batch (100% Alt section) |
| Alt Reduced | 5 | 3-leg Safe Over | ~20% below player L5 avg |
| **TOTAL** | **29** | | |

---

## Issues to Fix

### Issue 1: Parlay Card Shows Too Much Stats Per Leg
**Current:** Each leg row shows `"player name | direction line stat | 72% · -245"`
**Want:** Each leg row shows ONLY `"player name | direction line stat"`
**Rule:** Win % ONLY in the banner header at top of card. Nothing else next to player names.

### Issue 2: Overs Not Showing in Parlays
**Root cause:** `build_over_parlays` uses `model_prob >= 0.56` filter. Overs with lower hit rates (0.52–0.55) never show because model_prob ends up around 0.52–0.55 range. Unders only enter the pool at `hit_rate >= 0.62` (high bar), so their model_prob is systematically higher.
**Fix:** Lower over threshold to `0.52` (matching `_OVER_MIN_HIT_RATE`).

### Issue 3: Duplicate Parlays / Player Repetition
**Current tracker:** Tracks `(player, stat)` pairs — allows LaMelo to appear in over_PTS parlay #1, #2, then ALSO over_AST parlay #1, #2 = 4 total.
**User wants:** Max 2 parlays per player TOTAL across ALL stats and types.
**Fix:** Change `_DiversityTracker._key` to use only `player.lower()` — no stat component.

### Issue 4: Parlay Counts
**Current:** 1x ML, 2x alt (10-leg), 5x over, 3x under, 10x reduced, 3x defense
**Want:** 10x over, 3x ML, 3x total, 3x spread, 5x alt-overs, 5x alt-reduced = 29

### Issue 5: 100% Alt Section Needs Different Content
**Current:** 2x 10-leg alt line parlays
**Want:** 5x 3-leg overs (from best props) + 5x 3-leg reduced-avg

---

## Architecture

### Files to Modify
| File | Changes |
|------|---------|
| `utils/parlay_builder.py` | Fix tracker, fix over threshold, add ML trio / spread / total builders, update orchestrator |
| `utils/props_cache.py` | Pass spread + model_total data into game_predictions |
| `dashboard/app.py` | Strip leg stats from parlay cards, update `_create_parlays_section` layout |

---

## Part 1: `utils/parlay_builder.py`

### 1.1 Fix `_DiversityTracker._key` (Player-Only Tracking)

```python
# BEFORE
def _key(self, player: str, stat: str, direction: str = "") -> tuple:
    return (player.lower(), stat.upper())

# AFTER — track by player name only
def _key(self, player: str, stat: str = "", direction: str = "") -> tuple:
    return (player.lower(),)
```

**Why:** User wants max 2 parlays per player total, regardless of stat type.

### 1.2 Fix `build_over_parlays` (Lower Threshold)

```python
# BEFORE
overs = [
    p for p in props
    if p.get("direction", "").lower() == "over"
    and (p.get("model_prob") or 0) >= 0.56
    and p.get("model_over_odds") is not None
]

# AFTER
overs = [
    p for p in props
    if p.get("direction", "").lower() == "over"
    and ((p.get("model_prob") or 0) >= 0.52 or p.get("hit_rate", 0) >= 0.52)
    and p.get("model_over_odds") is not None
]
```

Change `n_parlays=5` → `n_parlays=15` (10 for main section + 5 for alt section).

### 1.3 Add `build_ml_trio` (3 Distinct ML Parlays)

```python
def build_ml_trio(
    game_predictions: list[dict],
    tracker: _DiversityTracker,
) -> list[dict]:
    """Build 3 distinct 3-leg ML parlays from today's games.

    Picks the top 9 qualifying games (HIGH/MEDIUM confidence),
    assigns them to 3 non-overlapping groups of 3.
    Returns 1–3 parlays depending on how many qualify.
    """
    CONF_TO_PROB = {"HIGH": 0.78, "MEDIUM": 0.62}
    candidates = []
    for g in game_predictions:
        conf  = g.get("winner_confidence", "LOW")
        prob  = CONF_TO_PROB.get(conf)
        winner = g.get("winner_pick")
        if prob is None or not winner:
            continue
        home = g.get("home", "")
        away = g.get("away", "")
        if winner != home:
            prob = max(0.52, 1.0 - prob)
        candidates.append({
            "player": f"{away} @ {home}",
            "stat": "ML", "direction": "ML", "pick": winner,
            "confidence": conf,
            "win_prob": round(prob * 100, 1),
            "model_odds": _prob_to_american(prob),
            "label": f"{winner} ML ({conf})",
            "game": f"{away} @ {home}",
        })

    # Sort by confidence descending
    candidates.sort(key=lambda x: -x["win_prob"])

    parlays = []
    for i in range(3):
        group = candidates[i * 3 : i * 3 + 3]
        if len(group) < 3:
            break
        parlays.append({
            "label": f"Moneyline Parlay #{i + 1}",
            "type": "ml",
            "legs": group,
            "odds": parlay_odds(group),
        })
    return parlays
```

### 1.4 Add `build_spread_parlays` (3x Game Spread Parlays)

```python
def build_spread_parlays(
    game_predictions: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 3,
) -> list[dict]:
    """Build 3x 3-leg spread (ATS) parlays from game predictions.

    Requires game_predictions to include 'spread' key (home line, negative = home favored).
    HOME covers if spread is between -3 and -9 (moderate home favorite).
    AWAY covers if spread is +3 to +9 (moderate underdog).
    """
    SPREAD_TO_PROB = {}   # Maps abs(spread) → estimated cover probability

    def _spread_prob(spread_val: float) -> float:
        abs_s = abs(spread_val)
        # Based on Massey-Peabody: prob ≈ 0.50 + abs_spread / 25
        return min(0.72, max(0.52, 0.50 + abs_s / 25.0))

    candidates = []
    for g in game_predictions:
        spread = g.get("spread")
        if spread is None:
            continue
        home = g.get("home", "")
        away = g.get("away", "")
        abs_s = abs(spread)
        if abs_s < 1.5:   # pick is too close to call
            continue
        if spread < 0:    # home favored
            pick_team = home
            prob = _spread_prob(spread)
        else:             # away favored (home is underdog)
            pick_team = away
            prob = _spread_prob(spread)

        candidates.append({
            "player": f"{away} @ {home}",
            "stat": "ATS", "direction": "ATS",
            "pick": pick_team,
            "win_prob": round(prob * 100, 1),
            "model_odds": _prob_to_american(prob),
            "label": f"{pick_team} ATS ({'+' if spread > 0 else ''}{spread:.1f})",
            "game": f"{away} @ {home}",
        })

    candidates.sort(key=lambda x: -x["win_prob"])

    parlays = []
    for i in range(n_parlays):
        group = candidates[i * 3 : i * 3 + 3]
        if len(group) < 3:
            break
        parlays.append({
            "label": f"Spread Parlay #{i + 1}",
            "type": "spread",
            "legs": group,
            "odds": parlay_odds(group),
        })
    return parlays
```

### 1.5 Add `build_total_parlays` (3x Game Total O/U Parlays)

```python
def build_total_parlays(
    game_predictions: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 3,
    default_nba_total: float = 220.0,
) -> list[dict]:
    """Build 3x 3-leg game total (over/under) parlays.

    Requires game_predictions to include 'model_total' key.
    Compares model_total vs default_nba_total (220) or actual O/U line if provided.
    """
    candidates = []
    for g in game_predictions:
        model_total = g.get("model_total")
        ou_line = g.get("total_line", default_nba_total)  # actual O/U if fetched, else 220
        if model_total is None:
            continue

        home = g.get("home", "")
        away = g.get("away", "")
        diff = model_total - ou_line

        if abs(diff) < 3.0:   # not enough edge
            continue

        direction = "Over" if diff > 0 else "Under"
        prob = min(0.70, max(0.52, 0.52 + abs(diff) / 20.0))

        candidates.append({
            "player": f"{away} @ {home}",
            "stat": "TOTAL", "direction": direction,
            "model_total": round(model_total, 1),
            "ou_line": ou_line,
            "win_prob": round(prob * 100, 1),
            "model_odds": _prob_to_american(prob),
            "label": f"{direction} {ou_line} (model: {model_total:.0f})",
            "game": f"{away} @ {home}",
        })

    candidates.sort(key=lambda x: -x["win_prob"])

    parlays = []
    for i in range(n_parlays):
        group = candidates[i * 3 : i * 3 + 3]
        if len(group) < 3:
            break
        parlays.append({
            "label": f"Game Totals Parlay #{i + 1}",
            "type": "totals",
            "legs": group,
            "odds": parlay_odds(group),
        })
    return parlays
```

### 1.6 Update `build_all_parlays` Orchestrator

```python
def build_all_parlays(
    props: list[dict],
    alt_lines: list[dict],
    game_predictions: list[dict],
) -> dict:
    tracker = _DiversityTracker(max_uses=2)

    # 10 main overs + 5 alt overs = 15 total overs
    all_overs = build_over_parlays(props, tracker, n_parlays=15, n_legs=3)
    main_overs = all_overs[:10]
    alt_overs  = all_overs[10:]

    # Alt reduced (5)
    reduced = build_reduced_avg_parlays(props, tracker, n_parlays=5, n_legs=3)

    # Game-level parlays (3 + 3 + 3)
    ml_parlays     = build_ml_trio(game_predictions, tracker)
    spread_parlays = build_spread_parlays(game_predictions, tracker, n_parlays=3)
    total_parlays  = build_total_parlays(game_predictions, tracker, n_parlays=3)

    total = (
        len(main_overs) + len(alt_overs) + len(reduced)
        + len(ml_parlays) + len(spread_parlays) + len(total_parlays)
    )

    return {
        "over":        main_overs,     # 10x main section
        "ml":          ml_parlays,     # 3x
        "spread":      spread_parlays, # 3x
        "totals":      total_parlays,  # 3x
        "alt_over":    alt_overs,      # 5x (100% Alt section)
        "reduced":     reduced,        # 5x (100% Alt section)
        # KEPT for backward compat (grading tracker, etc.)
        "alt":         [],
        "under":       [],
        "defense":     [],
        "total_count": total,
    }
```

**Remove from output:** `under`, `defense`, `alt` (still keep the functions but exclude from UI).

---

## Part 2: `utils/props_cache.py`

### 2.1 Extend `game_predictions_for_parlay`

Add `spread` and `model_total` to each game prediction entry:

```python
# After computing home_win_prob from spread:
game_predictions_for_parlay.append({
    "home": home_team,
    "away": away_team,
    "winner_pick": winner,
    "winner_confidence": conf,
    "spread": spread,         # ADD: home spread line (neg = home favored)
    "model_total": None,      # Will be filled if game predictor runs
})
```

**Note:** `model_total` will remain None unless we run `GamePredictor.enrich_picks` per game. As a first pass, `model_total=None` means total parlays won't build if no data — graceful empty state. Can be enhanced later to call the predictor.

---

## Part 3: `dashboard/app.py`

### 3.1 Strip Stats From Parlay Leg Rows

**File:** `dashboard/app.py`, function `_parlay_card()` (~line 1789–1810)

**Remove** the last `html.Span` from each leg row — the one that shows `f"{leg_wp:.0f}% · {leg_odds_str}"`:

```python
# BEFORE
leg_rows.append(html.Div([
    html.Span(li_icon, ...),
    html.Span(leg.get("player", ""), ...),
    html.Span(prop_text, ...),
    html.Span(f"{leg_wp:.0f}% · {leg_odds_str}", ...),   # ← REMOVE THIS
], ...))

# AFTER — player name + stat only
leg_rows.append(html.Div([
    html.Span(li_icon, ...),
    html.Span(leg.get("player", ""), ...),
    html.Span(prop_text, ...),
], ...))
```

Win probability stays in the header (it's already there via `html.Span(f"{win_prob}% win", ...)`).

### 3.2 Update `_create_parlays_section` Layout

**New section order:**

```
Section A: Props OVER Parlays (10)  ← was 5, now 10
Section B: Moneyline Parlays (3)    ← was 1, now 3
Section C: Spread Parlays (3)       ← NEW
Section D: Game Totals Parlays (3)  ← NEW
Section E: 100% Alt — Best Overs (5)   ← replaces 10-leg alts
Section F: 100% Alt — Reduced Lines (5) ← was 10, now 5
```

**Add to `TYPE_COLORS`:**
```python
"spread":  ("#f97316", "#431407"),   # orange
"totals":  ("#0ea5e9", "#0c4a6e"),   # sky blue
"alt_over": ("#22c55e", "#14532d"),  # green (same as over)
```

**Add to `TYPE_ICONS`:**
```python
"spread": "ATS", "totals": "O/U", "alt_over": "↑",
```

**Section rendering:**
```python
# Section A: Main Overs (10)
over_list = parlays.get("over", [])
cards.append(_section_header(f"Props OVER Parlays  ({len(over_list)})", "..."))
cards.append(_two_col_grid([_parlay_card(p) for p in over_list]))

# Section B: Moneyline (3)
ml_list = parlays.get("ml", [])
cards.append(_section_header(f"Moneyline Parlays  ({len(ml_list)})", "..."))
cards.append(_two_col_grid([_parlay_card(p) for p in ml_list]))

# Section C: Spread (3)
spread_list = parlays.get("spread", [])
if spread_list:
    cards.append(_section_header(f"Spread Parlays  ({len(spread_list)})", "..."))
    cards.append(_two_col_grid([_parlay_card(p) for p in spread_list]))

# Section D: Game Totals (3)
totals_list = parlays.get("totals", [])
if totals_list:
    cards.append(_section_header(f"Game Totals Parlays  ({len(totals_list)})", "..."))
    cards.append(_two_col_grid([_parlay_card(p) for p in totals_list]))

# Section E: 100% Alt — Best Overs (5)
alt_over_list = parlays.get("alt_over", [])
cards.append(_section_header("100% Alt — Best Over Parlays", "..."))
cards.append(_two_col_grid([_parlay_card(p) for p in alt_over_list]))

# Section F: 100% Alt — Reduced Lines (5)
reduced_list = parlays.get("reduced", [])
cards.append(_section_header("100% Alt — Reduced Lines", "..."))
cards.append(_two_col_grid([_parlay_card(p) for p in reduced_list]))
```

**Remove sections:** Under parlays, Defense parlays, old 10-leg Alt parlays.

---

## Implementation Steps

### Step 1 — `utils/parlay_builder.py`: Fix tracker key
- Change `_DiversityTracker._key` to return `(player.lower(),)` only
- **Why:** Enforces max 2 appearances per player globally

### Step 2 — `utils/parlay_builder.py`: Fix overs threshold
- Change `build_over_parlays` filter from `>= 0.56` to `>= 0.52`
- Change `n_parlays=5` → `n_parlays=15`
- **Why:** Overs were being excluded by too-strict model_prob gate

### Step 3 — `utils/parlay_builder.py`: Add `build_ml_trio`
- Replace `build_ml_parlay` (1 parlay) with `build_ml_trio` (3 parlays)
- Group top 9 games into non-overlapping trios
- Keep old `build_ml_parlay` function to avoid import errors

### Step 4 — `utils/parlay_builder.py`: Add `build_spread_parlays`
- New function, uses `spread` field from game_predictions
- Generates 3 distinct 3-leg spread parlays

### Step 5 — `utils/parlay_builder.py`: Add `build_total_parlays`
- New function, uses `model_total` field from game_predictions
- Falls back gracefully if model_total is None

### Step 6 — `utils/parlay_builder.py`: Change `build_reduced_avg_parlays` n from 10 to 5

### Step 7 — `utils/parlay_builder.py`: Update `build_all_parlays` orchestrator
- Wire in all new functions
- Return new keys: `over`, `ml`, `spread`, `totals`, `alt_over`, `reduced`
- Keep empty lists for `alt`, `under`, `defense` for backward compat

### Step 8 — `utils/props_cache.py`: Add spread to game_predictions
- Add `spread` key to each game_predictions_for_parlay entry

### Step 9 — `dashboard/app.py`: Strip leg stats from parlay cards
- Remove `html.Span(f"{leg_wp:.0f}% · {leg_odds_str}", ...)` from `_parlay_card()` leg rows
- Remove `leg_wp` and `leg_odds_str` from the row layout

### Step 10 — `dashboard/app.py`: Reshape `_create_parlays_section`
- Add `"spread"`, `"totals"`, `"alt_over"` to `TYPE_COLORS` and `TYPE_ICONS`
- Rewrite sections: Over(10) → ML(3) → Spread(3) → Totals(3) → Alt-Over(5) → Reduced(5)
- Remove Under, Defense, 10-leg Alt sections

### Step 11 — Verify
- `python3 -c "from utils.parlay_builder import build_all_parlays; print('ok')"` — no errors
- Run app: `python3 dashboard/app.py`
- Navigate to Parlays tab → verify 6 sections render
- Check parlay leg rows have NO win percentages or hit rate stats
- Confirm no player appears in more than 2 parlays (sample check)

---

## Risk Table

| Risk | Mitigation |
|------|------------|
| Not enough over props at 0.52 threshold | Already accepted: hit_rate >= 0.52 is the original entry bar |
| Only 5–8 qualifying games for ML/spread/total parlays | Graceful: show what's available, skip section if < 3 legs |
| Tracker change breaks backward compat | No external code calls `_key` directly — it's private |
| `model_total` always None (no predictor data) | Totals section shows "Not enough..." — expected fallback |
| 15x over parlays exhausts the player pool | Tracker still enforces 2-per-player; worst case fewer than 15 get built |
| Removing Under/Defense sections breaks grading | Grading uses saved `parlays_history.json` which has frozen data — no impact |

---

## SESSION_ID
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
