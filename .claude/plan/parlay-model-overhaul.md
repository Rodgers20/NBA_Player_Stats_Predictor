# Implementation Plan: Parlay Model Overhaul

## Problem Summary (from screenshot)

The current parlay tab shows 6 parlays but:
1. **Duplicate parlays** — #1 and #2 are identical, #3/#4 identical, #5/#6 identical
2. **Absurd odds / 0.4% win probability** — parlays showing +23562 are essentially lottery tickets, not bets
3. **No star players** — Brandon Miller, Moussa Diabate, Donovan Clingan, Toumani Camara dominate. Zero star anchors.
4. **Per-leg quality too low** — BLK 1.5, FG3M 2.5, STL 0.5 for bench players — these are near coin-flips and inflate odds to absurd levels
5. **Wrong stat type mix** — BLK/STL/FG3M legs don't belong in general OVER parlays

## Root Causes

### Bug 1: Duplicate parlays
The greedy builder reuses the same limited qualifying pool across N parlay slots. When only 5-6 props pass the filter, each parlay gets the same 4 legs. No frozenset deduplication exists on the output.

### Bug 2: Per-leg probability floor too low
`build_over_parlays` filters at `model_prob >= 0.52 OR hit_rate >= 0.52` — this is a coin-flip bar. Longshot props (e.g., Deni Avdija Over 17.5 PTS) can enter via hit_rate bypass while their `model_over_odds` is priced at +300, making the compound parlay odds absurd.

### Bug 3: No minimum joint win probability gate
`parlay_odds()` computes win_prob but the result is never checked before emitting the parlay. A 4-leg parlay at 0.4% should be immediately discarded.

### Bug 4: No star player requirement
The pool is sorted by `model_prob` descending with no role weighting. Rotation/bench players with moderate probabilities crowd out stars entirely.

### Bug 5: BLK/STL/FG3M in main OVER parlays
Defense stat props (BLK, STL) belong only in defense-specific parlays. FG3M belongs only in shooter-specific parlays. Including them in main OVER parlays dilutes the star content and produces garbage-time bench combinations.

## Implementation Plan

### Task Type
- [x] Backend (parlay_builder.py)
- [x] Minimal frontend update (app.py display — show win_prob prominently)

---

### Step 1: Add global deduplication to `_build_prop_parlays`

**File**: `utils/parlay_builder.py` — `_build_prop_parlays()` (~L401)

Add a `emitted_sets` frozenset tracker across all parlay iterations:

```python
emitted_sets: set[frozenset] = set()

for parlay_idx in range(n_parlays):
    legs = []
    ...build legs...

    leg_key = frozenset((l["player"], l["stat"]) for l in legs)
    if leg_key in emitted_sets:
        continue   # skip exact duplicate
    emitted_sets.add(leg_key)
    parlays.append(...)
```

### Step 2: Raise per-leg probability floor

**File**: `utils/parlay_builder.py` — `build_over_parlays()` (~L281)

Change filter from:
```python
(p.get("model_prob") or 0) >= 0.52 or p.get("hit_rate", 0) >= 0.52
```
To:
```python
(p.get("model_prob") or p.get("hit_rate", 0)) >= 0.62
and p.get("stat", p.get("stat_type", "")) in ("PTS", "AST", "REB")  # No BLK/STL/FG3M in main
```

**Also**: Remove the `or hit_rate >= 0.52` bypass entirely. Both conditions must use the primary probability.

### Step 3: Add minimum joint win probability gate

**File**: `utils/parlay_builder.py` — `_build_prop_parlays()` (~L456)

After computing `odds = parlay_odds(legs)`, add:
```python
# Minimum joint win probability: 6% for 4-leg, 10% for 3-leg
min_win_prob = 6.0 if len(legs) >= 4 else 10.0
if odds["win_prob"] < min_win_prob:
    continue  # don't emit — too speculative
```

### Step 4: Star player anchor requirement

**File**: `utils/parlay_builder.py` — `build_over_parlays()` + `_build_prop_parlays()`

Two changes:

**4a**: In `build_over_parlays`, sort the pool with star players first:
```python
overs.sort(key=lambda x: (
    x.get("role", "bench") not in ("star", "starter"),  # stars first
    -(x.get("model_prob") or 0)
))
```

**4b**: In `_build_prop_parlays`, enforce at least 1 star/starter in each parlay:
```python
# After building legs, check if it has at least 1 star or starter
roles_in_parlay = [l.get("role", "bench") for l in legs]
if not any(r in ("star", "starter") for r in roles_in_parlay):
    continue  # skip all-bench parlay
```

### Step 5: Exclude BLK/STL/FG3M from main OVER parlays

Already partially addressed in Step 2. Also add explicit exclusion:

**File**: `utils/parlay_builder.py` — `build_over_parlays()` filter

```python
_MAIN_PARLAY_STATS = {"PTS", "AST", "REB"}

overs = [
    p for p in props
    if p.get("direction", "").lower() == "over"
    and (p.get("stat_type") or p.get("stat", "")) in _MAIN_PARLAY_STATS
    and (p.get("model_prob") or p.get("hit_rate", 0)) >= 0.62
    and p.get("model_over_odds") is not None
]
```

### Step 6: Fix `build_all_parlays` — reduce n_parlays when pool is small

**File**: `utils/parlay_builder.py` — `build_all_parlays()` (~L1069)

Instead of always requesting 15 over parlays, cap at what the pool can support:
```python
pool_size = len([p for p in props if (p.get("model_prob") or 0) >= 0.62])
# Can realistically build ceil(pool_size / n_legs) unique parlays
max_unique = max(1, pool_size // 3)
n_main_overs = min(6, max_unique)   # cap at 6, never 15

all_overs = build_over_parlays(props, tracker, n_parlays=n_main_overs, n_legs=3)
```

### Step 7: Dashboard — show win_prob badge prominently + filter 0% parlays

**File**: `dashboard/app.py` — parlay card renderer

1. If `win_prob < 1.0`, don't render the parlay card at all
2. Show win_prob as a color-coded badge:
   - `>= 15%` → green
   - `8-15%` → yellow
   - `< 8%` → red (but still show if passes the gate)

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `utils/parlay_builder.py:L281-299` | Modify | Raise filter floor, exclude BLK/STL/FG3M |
| `utils/parlay_builder.py:L401-466` | Modify | Add dedup, min win_prob gate, star anchor |
| `utils/parlay_builder.py:L1069-1088` | Modify | Cap n_parlays to pool size |
| `utils/parlay_builder.py:L267-299` | Modify | Sort stars first |
| `dashboard/app.py` | Modify | Filter 0% parlays, color-coded win_prob badge |

---

## Expected Outcome

- **3–4 unique parlays** instead of 6 duplicates
- **Win probability**: 8–20% range (realistic, not lottery odds)
- **Star players anchoring** every parlay (LaMelo, Kawhi, Giannis, Jokic etc.)
- **Clean stat mix**: PTS/AST/REB legs only in main parlays
- **BLK/STL**: stay in Defense Parlays section only

---

### SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (Claude-only plan)
- GEMINI_SESSION: N/A
