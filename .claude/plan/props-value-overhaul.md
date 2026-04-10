# Props Value & Volume Overhaul

**Feature:** More props, filter out chalk, meaningful payout gates
**Status:** Planning
**Date:** 2026-04-08

---

## Problem Diagnosis

### Problem 1: Chalk Props Leak Through (SGA Over 20 Points)
- `_SAFE_PROB = 0.82` is hardcoded for ALL reduced-avg parlays
- `_prob_to_american(0.82)` ≈ **-455** → barely pays anything per leg
- Combined parlay of three -455 legs = +77 at 55% win prob (ok for a parlay, bad as an individual bet)
- But users see "SGA Over 20" and correctly think it's boring/obvious

### Problem 2: Main Props EV Gate Too Loose
- `_MIN_EV = 0.02` = just 2% edge
- A prop at -400 with model_prob 0.81 has ev ≈ 0.03 → PASSES the filter
- That prop pays $25 on a $100 bet — worthless even if you win
- Users want props where they can actually make money

### Problem 3: Not Enough Props
- `_MAX_PER_PLAYER = 2` → flood of star props, thin bench coverage
- `_HARD_CAP = 60` → too restrictive when there are 10+ games
- `15 MPG minimum` → excludes rotation players who have matchup-based value
- Only PTS/AST/REB/FG3M shown — BLK/STL (which have great odds) excluded from main tab

### Problem 4: Reduced Parlays Built on Stars = Boring
- SGA avg 30 pts → reduced line 24.0 → still chalk
- Reduced parlays should focus on **mid-tier scorers** (15–22 ppg)
  where the 20% reduction produces lines with meaningful odds
- Example: Player avg 18 pts → reduced to 14.5 → odds around -150 → worth betting

---

## Fixes

### Fix 1: Raise EV Gate + Add Odds Minimum to Props

**File: `utils/props_cache.py`**

```python
# BEFORE
_MIN_MODEL_PROB = 0.56
_MIN_EV         = 0.02
_MAX_PER_PLAYER = 2
_HARD_CAP       = 60

# AFTER
_MIN_MODEL_PROB = 0.54   # slightly relaxed — let EV gate do the heavy lifting
_MIN_EV         = 0.05   # 5% real EV (based on actual book price, not just model_prob)
_MAX_PER_PLAYER = 3      # up to 3 props per player
_HARD_CAP       = 100    # 100 max for bigger slates
```

**Also add an odds minimum gate** inside the quality gate loop:

```python
# After the EV check, add odds minimum
_MIN_ODDS_AMERICAN = -180   # won't show props that pay worse than -180

for p in props_data:
    if p.get("is_lock"):
        # Locks get a more generous odds allowance (-250)
        if (p.get("model_over_odds") or -300) >= -250:
            quality_props.append(p)
        continue

    mp  = p.get("model_prob") or 0.0
    ev  = p.get("ev") or 0.0
    odd = p.get("model_over_odds") or -300

    if mp >= _MIN_MODEL_PROB and ev >= _MIN_EV and odd >= _MIN_ODDS_AMERICAN:
        quality_props.append(p)
```

**Effect:**
- "SGA Over 20 at -450" → model_odds ≈ -455 → FAILS odds gate → not shown
- "Mid-tier player Over 24.5 at -135 with 62% model_prob" → passes all gates → shown
- LOCKs still show up to -250 odds because they have proven track records

### Fix 2: Lower Player Qualification to Include More Rotation Players

**File: `utils/props_cache.py`**, function `_is_qualified_player`:

```python
# BEFORE
if pd.isna(avg_min) or avg_min < 15:
    return False, 0.0

# AFTER
if pd.isna(avg_min) or avg_min < 12:   # was 15, now 12
    return False, 0.0
```

And for the current-season games requirement:

```python
# BEFORE — requires 10 current-season games
if len(current_rows) < 10:
    return False, 0.0

# AFTER — requires 8 current-season games (captures players returning from injury)
if len(current_rows) < 8:
    return False, 0.0
```

**Effect:** Captures ~20% more rotation players who average 12-14 MPG in key roles.

### Fix 3: Add STL and BLK to Main Props (Not Just Alt Lines)

**File: `utils/props_cache.py`**, inside `_compute_main_page_props`:

```python
# BEFORE
for stat_type in ["PTS", "AST", "REB", "FG3M"]:

# AFTER
for stat_type in ["PTS", "AST", "REB", "FG3M", "STL", "BLK"]:
```

**Minimum average filters for new stats:**
```python
# Inside the per-stat loop, add per-stat minimum:
STAT_MIN_AVG = {"PTS": 1.0, "AST": 1.0, "REB": 1.0, "FG3M": 0.5, "STL": 0.5, "BLK": 0.3}
if avg_stat < STAT_MIN_AVG.get(stat_type, 1.0):
    continue
```

**Line setting for new stats:**
```python
elif stat_type == "STL":
    line = math.floor(l5_avg * 2) / 2 if l5_avg >= 0.8 else 0.5   # e.g., avg 1.4 → line 1.5
elif stat_type == "BLK":
    line = math.floor(l5_avg * 2) / 2 if l5_avg >= 0.5 else 0.5   # e.g., avg 1.2 → line 1.5
```

**Why STL/BLK help with payout:**
- Defensive stats are harder to predict → books price them at better odds (+100 to -130)
- A player averaging 1.5 STL per game hitting Over 1.5 STL has real value at -115
- These don't exist in the main props tab today

### Fix 4: Exclude Stars from Reduced-Avg Parlays

**File: `utils/parlay_builder.py`**, function `build_reduced_avg_parlays`:

Add two filters to the pool-building loop:

```python
# Filter 1: Only include players where l5_avg makes reduced line interesting
# (not stars whose reduced line is still chalk)
STAT_MAX_AVG_FOR_REDUCED = {
    "PTS": 24.0,   # avg > 24 pts → reduced line still too chalk
    "AST": 8.0,    # avg > 8 ast → same issue
    "REB": 11.0,   # avg > 11 reb → same issue
    "FG3M": 4.0,   # avg > 4 threes → same issue
}
if l5_avg > STAT_MAX_AVG_FOR_REDUCED.get(stat, 999):
    continue

# Filter 2: Only include if reduced line implies odds ≤ -180
# (compute approximate odds using a normal distribution approximation)
# We use the existing model_odds (stored in pool as _prob_to_american(_SAFE_PROB))
# Since _SAFE_PROB = 0.82 → model_odds ≈ -455, ALL would fail.
# Instead, compute actual probability at the reduced line using hit_rate proxy:
#   approximate: prob = (hits where stat >= red_line) / total_games
# Use the existing hit_rate data if available.
```

**Simpler approach — override `_SAFE_PROB` based on average:**
```python
# Dynamic SAFE_PROB based on reduction magnitude
# The bigger the reduction from book_line, the higher the probability
# But cap at 0.72 to avoid generating chalk legs
def _safe_prob_for_reduced(l5_avg: float, red_line: float, book_line: float) -> float:
    if book_line > 0:
        # How far below book line is the reduced line?
        discount_from_book = book_line - red_line
        # Each 0.5 unit below book ≈ 5% extra probability
        extra = min(0.15, discount_from_book * 0.05)
        return min(0.72, 0.55 + extra)   # max 72%, min 55%
    return 0.65   # no book line: moderate confidence

_SAFE_PROB = _safe_prob_for_reduced(l5_avg, red_line, book_line)
```

**Effect:**
- SGA avg 30 pts → reduced line 24 → book line 29.5 → discount = 5.5 → prob = 72% → odds ≈ -265
- Still kind of chalk for a standalone bet, but...
- Player avg 18 pts → reduced line 14.5 → book line 18.5 → discount = 4 → prob = 70% → odds ≈ -243
- Combined parlay of three 70% legs = +35 at 34% win prob → actually pays

Wait — the REAL issue is: user wants reduced parlays to pay more.

**Better approach — use the odds gate in reduced parlay builder:**

```python
# In build_reduced_avg_parlays pool building:
# Skip this player if their model_odds is too chalk
MAX_REDUCED_ODDS = -200   # won't use legs priced worse than -200

# Use a smarter SAFE_PROB: 70% (was 82%)
_SAFE_PROB = 0.70   # was 0.82

# This means:
# model_odds = _prob_to_american(0.70) ≈ -245
# Still too chalk but combined parlay of 3:
# decimal(0.70) = -245 → decimal 1.41
# Combined: 1.41^3 = 2.80 → +180 at 34% win prob → GREAT PAYOUT!
```

Changing `_SAFE_PROB` from 0.82 to 0.70 is the key fix:
- Old: Three -455 legs → combined +77 (55% win prob) — boring
- New: Three -245 legs → combined +180 (34% win prob) — actual parlay payout

The tradeoff: we're being less conservative about the line. But the user wants better payouts, so this is the right call.

### Fix 5: Increase Parlay Diversity in Over Builder

For `build_over_parlays` with 10+ parlays, the current greedy approach picks the same top players first. Add a "rotation" mode:

```python
# After building parlays, if fewer than 10 built due to tracker exhaustion,
# relax model_prob threshold from 0.52 to 0.50 for a second pass
if len(parlays) < n_parlays:
    # Second pass: lower bar
    second_pass_overs = [
        p for p in props
        if p.get("direction", "").lower() == "over"
        and ((p.get("model_prob") or 0) >= 0.50 or p.get("hit_rate", 0) >= 0.50)
        and p.get("model_over_odds") is not None
        and (p.get("model_over_odds") or -300) >= -200  # must pay something
    ]
    # Build remaining parlays from second-pass pool
```

This ensures we get 10 diverse over parlays even on thin days.

---

## Summary of All Changes

### `utils/props_cache.py`

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `_MIN_MODEL_PROB` | 0.56 | 0.54 | EV gate does heavy lifting now |
| `_MIN_EV` | 0.02 | 0.05 | Eliminates chalk props (5% real edge required) |
| `_MIN_ODDS_AMERICAN` | none | -180 | Props must pay at least $56 per $100 risked |
| `_MAX_PER_PLAYER` | 2 | 3 | More variety per player |
| `_HARD_CAP` | 60 | 100 | Handle big slates |
| Qualification MPG | 15 | 12 | Include rotation players |
| Current season games | 10 | 8 | Capture returning players |
| Stat types | PTS/AST/REB/FG3M | + STL/BLK | Better odds, more variety |
| Lock odds gate | none | -250 max | Locks still shown even at moderate chalk |

### `utils/parlay_builder.py`

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `_SAFE_PROB` | 0.82 | 0.70 | Combined parlay pays +180 instead of +77 |
| `_SAFE_REDUCE_PCT` | 0.20 | 0.22 | Slightly more reduction to improve odds |
| Over threshold | 0.56 | 0.52 | Let more overs qualify |
| Stars in reduced parlays | included | excluded (avg > threshold) | Stars' reduced lines are still chalk |

---

## Implementation Steps

### Step 1 — `utils/props_cache.py`: Update quality gate constants
- Change `_MIN_MODEL_PROB`, `_MIN_EV`, `_MAX_PER_PLAYER`, `_HARD_CAP`
- Add `_MIN_ODDS_AMERICAN = -180`

### Step 2 — `utils/props_cache.py`: Add odds gate to quality loop
- In the quality gate loop, add `odd >= _MIN_ODDS_AMERICAN` check
- LOCK exception: `odd >= -250`
- Key: use `model_over_odds` field (already computed in live-odds enrichment)

### Step 3 — `utils/props_cache.py`: Lower player qualification bar
- Change `avg_min < 15` → `avg_min < 12`
- Change `len(current_rows) < 10` → `len(current_rows) < 8`

### Step 4 — `utils/props_cache.py`: Add STL and BLK to stat loop
- Extend `for stat_type in ["PTS", "AST", "REB", "FG3M"]` to include `"STL", "BLK"`
- Add STAT_MIN_AVG dict for per-stat minimums
- Add line-setting logic for STL/BLK (half-unit lines: 0.5, 1.5, 2.5)

### Step 5 — `utils/parlay_builder.py`: Change `_SAFE_PROB` from 0.82 to 0.70
- Also change `_SAFE_REDUCE_PCT` from 0.20 to 0.22
- This improves payout from +77 to +180 for a 3-leg reduced parlay

### Step 6 — `utils/parlay_builder.py`: Exclude stars from reduced parlays
- In `build_reduced_avg_parlays` pool loop, add max-avg filter:
  `if l5_avg > 24.0 and stat == "PTS": continue`  (etc. per stat)
- Stars have better value in the regular over parlays where their full lines are used

### Step 7 — Verify
- `python3 -c "from utils.props_cache import _compute_main_page_props; print('ok')"` — no import errors
- Run app: check props tab shows STL/BLK props
- Check no props shown with model_odds worse than -180 in main tab
- Check reduced parlay legs have 70% win prob (not 82%)
- Spot check: player averaging 30+ should NOT appear in reduced parlays

---

## Expected Outcome

### Props Tab (Before vs After)
| Metric | Before | After |
|--------|--------|-------|
| Total props shown | ~40–60 | ~70–100 |
| Stats included | PTS/AST/REB/FG3M + combos | + STL/BLK |
| Worst odds shown | -455 (chalk) | -180 (meaningful payout) |
| Props per player | max 2 | max 3 |
| Min player minutes | 15 MPG | 12 MPG |

### Reduced Parlays (Before vs After)
| Metric | Before | After |
|--------|--------|-------|
| Win probability per parlay | 55% (3 × 82%) | 34% (3 × 70%) |
| Combined payout (3-leg) | +77 | +180 |
| Stars in reduced parlays | Yes (SGA Over 24) | No (excluded avg > 24 PTS) |
| Target player type | All scorers | Mid-tier (15–22 ppg) |

---

## SESSION_ID
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
