# Parlay Section Overhaul — Ultra Plan

**Feature:** New parlay section with 10 three-leg parlays, 10 best bets (single picks), 10 two-leg parlays
**Status:** Planning
**Date:** 2026-04-08

---

## Requirements

| Section | Count | Description |
|---------|-------|-------------|
| Best Bets | 10 single picks | Highest model-confidence individual props |
| 2-Leg Parlays | 10 parlays | Tight, high-win-prob pairs |
| 3-Leg Parlays | 10 parlays | Unified pool (not siloed over/under/defense) |

---

## Current State

### What exists in `parlay_builder.py`
- `build_ml_parlay()` → 1x 3-leg moneyline
- `build_alt_parlays()` → 2x 10-leg alt lines
- `build_over_parlays()` → 5x 3-leg overs only
- `build_under_parlays()` → 3x 3-leg unders only
- `build_defense_parlays()` → 3x 3-leg BLK/STL only
- `build_all_parlays()` → orchestrator returning `{ml, alt, over, under, defense, total_count}`

### What's missing
- No single-pick "best bets" concept
- No 2-leg parlays
- 3-leg parlays are siloed by direction (over vs under) instead of unified by confidence
- No unified ranking across all prop types for 3-leg selection

### Prop data shape (key fields for selection)
```python
{
    "player": str,
    "team": str,
    "opponent": str,
    "stat": str,           # "PTS", "AST", "REB", "FG3M", "STL", "BLK", combo
    "direction": str,      # "Over" or "Under"
    "line": float,
    "model_prob": float,   # 0-1, model's probability estimate (KEY SIGNAL)
    "ev": float,           # Expected value (0-1 scale)
    "edge": float,         # model_prob - implied_prob
    "is_lock": bool,       # hit_rate >= 80% AND n >= 5
    "hit_rate": float,     # historical hit rate vs current line
    "has_live_odds": bool,
    "live_over_price": int | None,
    "live_under_price": int | None,
    "model_over_odds": int,
    "model_under_odds": int,
    "blowout_risk": bool,
    "is_combo": bool,
}
```

---

## Architecture Plan

### Files to Modify
| File | Change Type | Scope |
|------|-------------|-------|
| `utils/parlay_builder.py` | Extend | Add 3 new build functions, update orchestrator |
| `dashboard/app.py` | Extend | Redesign `_create_parlays_section()` |
| `utils/props_cache.py` | None | No changes needed (already calls `build_all_parlays`) |

---

## Part 1: `utils/parlay_builder.py`

### 1.1 New function: `build_best_bets(props, n=10)`

**Purpose:** Select top 10 individual props with highest model confidence.

**Selection algorithm:**
```python
def build_best_bets(props, n=10):
    # Step 1: Filter eligible props
    eligible = [
        p for p in props
        if p.get("model_prob", 0) >= 0.63          # High confidence bar
        and not p.get("blowout_risk", False)         # Skip garbage time risk
        and not p.get("is_combo", False)             # Combos have noisier probs
    ]

    # Step 2: Sort — locks first, then by model_prob DESC, then by EV DESC
    eligible.sort(key=lambda p: (
        not p.get("is_lock", False),   # True first (locks bubble up)
        -p.get("model_prob", 0),
        -p.get("ev", 0),
    ))

    # Step 3: Deduplicate by player — max 1 pick per player
    seen_players = set()
    best_bets = []
    for p in eligible:
        player = p["player"]
        if player not in seen_players:
            seen_players.add(player)
            best_bets.append(p)
        if len(best_bets) == n:
            break

    # Step 4: If we don't have 10 at 0.63, relax to 0.58 and try again
    if len(best_bets) < n:
        fallback = [
            p for p in props
            if p.get("model_prob", 0) >= 0.58
            and not p.get("blowout_risk", False)
            and p["player"] not in seen_players
        ]
        fallback.sort(key=lambda p: (-p.get("model_prob", 0), -p.get("ev", 0)))
        for p in fallback:
            best_bets.append(p)
            if len(best_bets) == n:
                break

    return best_bets[:n]
```

**Output:** List of raw prop dicts (not parlay format — display directly as pick cards).

**Confidence labels to attach at display time:**
| model_prob | Label |
|-----------|-------|
| >= 0.75 | "ELITE" |
| >= 0.68 | "STRONG" |
| >= 0.63 | "HIGH" |
| >= 0.58 | "SOLID" |

---

### 1.2 New function: `build_two_leg_parlays(props, tracker, n=10)`

**Purpose:** Build 10 two-legged parlays from highest-quality prop pool.

**Eligibility:** model_prob >= 0.60, has_live_odds preferred (fall back to model odds), no blowout_risk

**Pairing strategy (priority order):**
1. Lock + Lock (both legs have is_lock=True)
2. Lock + High-EV (one lock, one with ev >= 0.05)
3. High-EV + High-EV (both ev >= 0.05)
4. High-prob pairs (both model_prob >= 0.60)

**Algorithm:**
```python
def build_two_leg_parlays(props, tracker, n=10):
    # Build pool: model_prob >= 0.60
    pool = sorted(
        [p for p in props
         if p.get("model_prob", 0) >= 0.60
         and not p.get("blowout_risk", False)],
        key=lambda p: (-int(p.get("is_lock", False)), -p.get("model_prob", 0))
    )

    parlays = []
    used_pairs = set()  # frozenset of (player, stat) tuples to avoid duplicate parlays

    for i, leg_a in enumerate(pool):
        if len(parlays) == n:
            break
        key_a = (leg_a["player"], leg_a["stat"])
        if tracker.count(key_a) >= 2:
            continue

        for leg_b in pool[i+1:]:
            # No same player
            if leg_b["player"] == leg_a["player"]:
                continue

            key_b = (leg_b["player"], leg_b["stat"])
            if tracker.count(key_b) >= 2:
                continue

            pair = frozenset([key_a, key_b])
            if pair in used_pairs:
                continue

            # Build parlay
            legs = [_to_parlay_leg(leg_a), _to_parlay_leg(leg_b)]
            odds = parlay_odds(legs)

            # Minimum combined win probability: 32% (e.g., 60% x 60% = 36%)
            if odds["win_prob"] < 32.0:
                continue

            used_pairs.add(pair)
            tracker.record(key_a)
            tracker.record(key_b)

            parlays.append({
                "label": f"2-Leg Parlay #{len(parlays)+1}",
                "type": "two_leg",
                "legs": legs,
                "odds": odds,
            })
            break

    return parlays
```

**Win probability targets:**
- Ideal: 36–49% (two 60–70% legs)
- Floor: 32% (won't show below this)

---

### 1.3 New function: `build_three_leg_parlays(props, alt_lines, tracker, n=10)`

**Purpose:** Build 10 unified 3-leg parlays from best available props (direction-agnostic). Better than current siloed over/under approach.

**Pool construction:**
```python
# Combine regular props + alt lines into one ranked pool
# For alt lines: convert to prop-like dict with model_prob from window-to-prob mapping
# Pool eligibility: model_prob >= 0.56, no blowout_risk
```

**Greedy triplet selection:**
```python
def build_three_leg_parlays(props, alt_lines, tracker, n=10):
    # Step 1: Build unified pool
    pool = []

    # Regular props
    for p in props:
        if p.get("model_prob", 0) >= 0.56 and not p.get("blowout_risk", False):
            pool.append({
                "source": "prop",
                "player": p["player"],
                "stat": p["stat"],
                "direction": p["direction"],
                "line": p["line"],
                "model_prob": p["model_prob"],
                "ev": p.get("ev", 0),
                "is_lock": p.get("is_lock", False),
                "model_odds": p["model_over_odds"] if p["direction"] == "Over" else p["model_under_odds"],
                "win_prob": p["model_prob"] * 100,
                **p,
            })

    # Alt line props (convert streak → probability)
    for a in alt_lines:
        window = a.get("window", 5)
        prob = _WINDOW_TO_PROB.get(window, 0.72)  # existing mapping
        pool.append({
            "source": "alt",
            "player": a["player"],
            "stat": a["stat"],
            "direction": "Over",
            "line": float(a["threshold"]),
            "model_prob": prob,
            "ev": prob - 0.52,  # estimated EV vs -110 implied
            "is_lock": True,    # alt lines are by definition 100% in window
            "model_odds": _prob_to_american(prob),
            "win_prob": prob * 100,
            **a,
        })

    # Step 2: Sort pool by composite score
    pool.sort(key=lambda p: (
        -int(p.get("is_lock", False)),
        -p["model_prob"],
        -p.get("ev", 0),
    ))

    # Step 3: Build parlays greedily
    parlays = []
    used_triplets = set()

    for anchor in pool:
        if len(parlays) == n:
            break
        key_anchor = (anchor["player"], anchor["stat"])
        if tracker.count(key_anchor) >= 2:
            continue

        # Find best 2 companions for this anchor
        companions = []
        for candidate in pool:
            if candidate["player"] == anchor["player"]:
                continue
            if any(c["player"] == candidate["player"] for c in companions):
                continue
            key_c = (candidate["player"], candidate["stat"])
            if tracker.count(key_c) >= 2:
                continue
            companions.append(candidate)
            if len(companions) == 2:
                break

        if len(companions) < 2:
            continue

        triplet_key = frozenset([
            key_anchor,
            (companions[0]["player"], companions[0]["stat"]),
            (companions[1]["player"], companions[1]["stat"]),
        ])
        if triplet_key in used_triplets:
            continue

        legs = [
            _to_parlay_leg(anchor),
            _to_parlay_leg(companions[0]),
            _to_parlay_leg(companions[1]),
        ]
        odds = parlay_odds(legs)

        # Floor: 20% combined win probability (e.g., 3 x 58% = 19.5%)
        if odds["win_prob"] < 18.0:
            continue

        used_triplets.add(triplet_key)
        tracker.record(key_anchor)
        tracker.record((companions[0]["player"], companions[0]["stat"]))
        tracker.record((companions[1]["player"], companions[1]["stat"]))

        parlays.append({
            "label": f"3-Leg Parlay #{len(parlays)+1}",
            "type": "three_leg",
            "legs": legs,
            "odds": odds,
        })

    return parlays
```

**Win probability targets:**
- Ideal: 22–35% (three 60–70% legs)
- Floor: 18% (won't show below this)

---

### 1.4 Update `build_all_parlays()` orchestrator

**New signature (backward-compatible additions):**
```python
def build_all_parlays(props, alt_lines, game_predictions):
    tracker = DiversityTracker()  # existing

    result = {
        # NEW sections
        "best_bets": build_best_bets(props, n=10),
        "two_leg": build_two_leg_parlays(props, tracker, n=10),
        "three_leg": build_three_leg_parlays(props, alt_lines, tracker, n=10),

        # KEEP existing (for alt-lines tab + backward compat)
        "ml": build_ml_parlay(game_predictions, tracker),
        "alt": build_alt_parlays(alt_lines, tracker, n_parlays=2, n_legs=10),
        "over": build_over_parlays(props, tracker, n_parlays=5, n_legs=3),
        "under": build_under_parlays(props, tracker, n_parlays=3, n_legs=3),
        "defense": build_defense_parlays(alt_lines, tracker, n_parlays=3, n_legs=3),

        "total_count": 0,  # calculated below
    }

    result["total_count"] = (
        len(result["best_bets"])
        + len(result["two_leg"])
        + len(result["three_leg"])
        + len(result["alt"])
        + len(result["over"])
        + len(result["under"])
        + len(result["defense"])
        + (1 if result["ml"] else 0)
    )

    return result
```

**Note on tracker ordering:** Run `best_bets` first (no tracker usage — single picks don't conflict), then `two_leg`, then `three_leg`. The existing ml/alt/over/under/defense run after and get remaining slots.

---

### 1.5 Helper: `_to_parlay_leg(prop_or_alt)`

Unified converter that normalizes both prop dicts and alt-line dicts into parlay leg format:

```python
def _to_parlay_leg(p):
    return {
        "player": p["player"],
        "stat": p.get("stat", p.get("stat_label", "?")),
        "direction": p.get("direction", "Over"),
        "line": float(p.get("line", p.get("threshold", 0))),
        "win_prob": p.get("model_prob", p.get("win_prob", 0.65)) * 100,
        "model_odds": p.get("model_odds", p.get("model_over_odds", -110)),
        "is_lock": p.get("is_lock", False),
        "team": p.get("team", ""),
        "opponent": p.get("opponent", ""),
    }
```

---

## Part 2: `dashboard/app.py` — `_create_parlays_section()`

### 2.1 New layout structure

```
Parlays Tab
├── Section A: "🔒 Top 10 Best Bets" (single picks grid)
│   └── 10 pick cards (2-column grid, compact)
├── Section B: "⚡ 2-Leg Parlays"
│   └── 10 parlay cards (2-column grid)
└── Section C: "🎯 3-Leg Parlays"
    └── 10 parlay cards (2-column grid)

[collapsed / secondary]
└── Section D: "Alt Lines Parlays" (existing 10-leg + ML)
```

### 2.2 Best Bet Card Design

```
┌─────────────────────────────────────────┐
│ 🔒 LOCK    ELITE 78%          EV +14%   │
│                                          │
│  LaMelo Ball · CHA vs BOS               │
│  POINTS   OVER  22.5                    │
│                                          │
│  Hit Rate: 90%   L5 Avg: 26.2           │
│  DraftKings: -135  Model: -220          │
└─────────────────────────────────────────┘
```

**Card color system:**
| Condition | Border/Accent Color |
|-----------|---------------------|
| is_lock + model_prob >= 0.70 | Gold `#FFD700` |
| model_prob >= 0.68 | Green `#22c55e` |
| model_prob >= 0.63 | Teal `#06b6d4` |
| Fallback | Default `#3a3d4a` |

**Confidence badge mapping:**
| model_prob | Badge text | Badge color |
|-----------|------------|-------------|
| >= 0.75 | ELITE | `#FFD700` |
| >= 0.68 | STRONG | `#22c55e` |
| >= 0.63 | HIGH | `#06b6d4` |
| >= 0.58 | SOLID | `#a78bfa` |

### 2.3 Parlay Card Design (shared for 2-leg and 3-leg)

```
┌──────────────────────────────────────────────────────┐
│  3-Leg Parlay #1              WIN PROB  Combined Odds │
│                                  28%     +285         │
│  ─────────────────────────────────────────────────── │
│  ✓ LaMelo Ball    PTS OVER 22.5    72% │ Model -245  │
│  ✓ Evan Mobley    REB OVER 9.5     68% │ Model -200  │
│  🔒 Jalen Johnson  PTS OVER 8.5   90% │ Model -850  │
│                                                       │
│              $100 → $385 payout                       │
└──────────────────────────────────────────────────────┘
```

**Leg icon:**
- `🔒` if is_lock=True
- `⚡` if model_prob >= 0.68
- `●` otherwise

**Win probability color:**
- >= 35%: Green
- >= 25%: Yellow
- < 25%: Muted gray

### 2.4 Section headers with summary stats

```python
# Best Bets header
f"🔒 Top 10 Best Bets  ({n_locks} Locks · Avg Confidence {avg_prob:.0%})"

# 2-Leg Parlays header
f"⚡ 2-Leg Parlays  (Avg Win Prob {avg_win:.1f}%)"

# 3-Leg Parlays header
f"🎯 3-Leg Parlays  (Avg Win Prob {avg_win:.1f}%)"
```

### 2.5 Graceful empty state

If fewer than 10 picks/parlays are generated (low-prop day), show what's available with a note:

```python
if len(best_bets) < 10:
    # Show available picks + gray placeholder cards for remaining slots
    # Label: "Only X high-confidence picks available today"
```

---

## Part 3: Confidence Thresholds Reference

| Use Case | model_prob | Notes |
|----------|-----------|-------|
| Best Bets (primary) | >= 0.63 | Relaxes to 0.58 if < 10 available |
| Best Bets (ELITE badge) | >= 0.75 | Gold highlighting |
| 2-Leg eligibility | >= 0.60 | Min combined win prob: 32% |
| 3-Leg eligibility | >= 0.56 | Min combined win prob: 18% |
| Lock classification | hit_rate >= 0.80 AND n >= 5 | Existing gate |

---

## Part 4: Implementation Steps

### Step 1 — `parlay_builder.py`: Add `build_best_bets()`
- Filter by model_prob >= 0.63 (relax to 0.58 if needed)
- Deduplicate by player (max 1 per player)
- Sort: locks first → model_prob DESC → EV DESC
- Return list of 10 prop dicts (raw, not wrapped in parlay format)

### Step 2 — `parlay_builder.py`: Add `_to_parlay_leg()` helper
- Normalize prop dict or alt-line dict to unified leg format
- Handle both `model_over_odds`/`model_under_odds` (props) and single `model_odds` (alts)

### Step 3 — `parlay_builder.py`: Add `build_two_leg_parlays()`
- Pool: model_prob >= 0.60, no blowout
- Priority pairs: lock+lock, lock+ev, ev+ev, high-prob+high-prob
- Enforce player diversity within parlay + tracker across parlays
- Min combined win prob: 32%
- Return 10 parlay dicts

### Step 4 — `parlay_builder.py`: Add `build_three_leg_parlays()`
- Unified pool: regular props + alt lines (converted to common format)
- Sort composite: lock → model_prob → EV
- Greedy anchor + 2 best companions (player-diverse)
- Min combined win prob: 18%
- Return 10 parlay dicts

### Step 5 — `parlay_builder.py`: Update `build_all_parlays()`
- Add `best_bets`, `two_leg`, `three_leg` keys
- Keep all existing keys for backward compatibility
- Order tracker usage: two_leg → three_leg → existing parlays

### Step 6 — `dashboard/app.py`: Update `_create_parlays_section()`
- Read `parlays["best_bets"]`, `parlays["two_leg"]`, `parlays["three_leg"]`
- Build Section A: Best Bets grid (10 pick cards, 2-col)
- Build Section B: 2-Leg Parlays grid (10 cards, 2-col)
- Build Section C: 3-Leg Parlays grid (10 cards, 2-col)
- Keep existing alt/ML parlays as collapsed Section D

### Step 7 — Verify
- Run `python3 dashboard/app.py` — no import/syntax errors
- Click Parlays tab — verify 3 sections render
- Check parlay count labels match actual data
- Test with low-prop day (< 10 eligible) — verify graceful fallback

---

## Key Design Decisions

1. **Best bets are raw props, not parlays** — display as single pick cards, not wrapped in parlay format. This lets us show hit_rate, live odds, and EV alongside the model prob.

2. **3-leg parlays are direction-agnostic** — mix overs and unders in the same parlay. This is intentional: the strongest legs regardless of direction produce the best risk-adjusted parlays.

3. **Alt lines feed into 3-leg parlays** — the alt line pool (100% streaks) provides high-prob anchors for 3-leg parlays, not just the existing 10-leg alt parlays.

4. **Tracker priority:** `two_leg` runs before `three_leg` which runs before legacy builders. This ensures the new sections get first access to the best players/props. Legacy sections (over/under/defense) consume the remainder.

5. **No tracker for best bets** — single picks don't need cross-parlay diversity tracking. One player can appear in a best bet AND in a parlay (they're separate bet types).

6. **Backward compatibility** — all existing keys (`ml`, `alt`, `over`, `under`, `defense`) remain in the output dict. The existing Alt Lines tab behavior is unchanged.

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Not enough high-prob props (< 10) | Relax thresholds progressively; show available count with note |
| Duplicate player in same parlay | Player-uniqueness check within each parlay build loop |
| Same player appearing in 5+ parlays | DiversityTracker caps at 2 appearances across all parlays |
| Very low combined win probability | Hard floor checks (32% for 2-leg, 18% for 3-leg) |
| Alt lines not available (no games) | Alt pool gracefully empty; 3-leg parlays fall back to props-only |
| app.py `_create_parlays_section` not receiving new keys | Defensive `.get("best_bets", [])` pattern throughout |

---

## SESSION_ID
- CODEX_SESSION: N/A (Claude-native plan, no external model sessions)
- GEMINI_SESSION: N/A

---

**Execute with:**
```
/ccg:execute .claude/plan/parlay-section-overhaul.md
```
