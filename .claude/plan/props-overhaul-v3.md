# Props System Overhaul v3 — Implementation Plan

**Files:** `utils/props_cache.py`, `utils/parlay_builder.py`
**Scope:** 7 discrete changes across both files, ordered by dependency.

---

## Change Index

| # | Requirement | File | Lines Affected |
|---|-------------|------|----------------|
| 1 | Overs only — remove Under props | `props_cache.py` | 486–491 |
| 2 | Garbage-time player exclusion (minutes std dev gate) | `props_cache.py` | 196–228 |
| 3 | Tiered star player prioritization | `props_cache.py` | 244–291, 706–719 |
| 4 | Value alt lines at 72% of L5 avg | `props_cache.py` | 346–355 |
| 5 | High-value stat focus (quality gate + value score) | `props_cache.py` | 682–719 |
| 6 | Exclude bench players from alt parlays | `parlay_builder.py` | 197–261 |
| 7 | 3x 10-leg alt parlays (was 2) | `parlay_builder.py` | 199–204 |

---

## Change 1 — Overs Only

### What
Remove all Under prop generation from the stat loop and the combo loop.

### Where
`props_cache.py` lines 486–491 (Under prop emission inside the stat loop):

```python
# REMOVE — lines 486–491:
# Under — high bar (62%) AND must make contextual sense
under_makes_sense = not (blowout_risk and role in ("rotation", "bench"))
if (hit_rate_under > _UNDER_MIN_HIT_RATE
        and hit_rate_under != hit_rate_over
        and under_makes_sense):
    props_data.append(_make_prop("Under", under_line, hit_rate_under, hits_under))
```

Also remove the `_UNDER_MIN_HIT_RATE` constant (line 156) since it is no longer used.

Also audit the `_make_prop` function (lines 400–480) and remove the Under-specific branches:
- Lines 415–416: `else:  # Under becomes more likely for resting starters`
- Lines 421–422: `else:  # Under for bench in blowout = nonsensical`
- Lines 447–451: `hr_home`/`hr_away` else-branch computing under hit rates for home/away splits

After removal, `_make_prop` should only compute Over split stats. Simplify to:

```python
# KEEP only Over split stats in _make_prop:
hr_home = (home_games[stat_type] >= bet_line).sum() / len(home_games) if not home_games.empty else 0
hr_away = (away_games[stat_type] >= bet_line).sum() / len(away_games) if not away_games.empty else 0
h_home  = (home_games[stat_type] >= bet_line).sum() if not home_games.empty else 0
h_away  = (away_games[stat_type] >= bet_line).sum() if not away_games.empty else 0
```

### Constants to remove
```python
# DELETE line 156:
_UNDER_MIN_HIT_RATE = 0.62
```

---

## Change 2 — Garbage-Time Player Exclusion

### What
Add a "garbage time" detector to `_is_qualified_player`. Players who have:
- avg minutes < 18 AND
- minutes standard deviation > 10 (high-variance role: only plays in blowouts)

...should be excluded entirely from the props pipeline.

### Where
`props_cache.py` — `_is_qualified_player` function, lines 196–228.

### Implementation

After the existing `avg_min < 12` check, add:

```python
# Garbage-time exclusion: high std dev + low avg = blowout-only player
# (e.g. Paul Reed, Jamal Cain who spike minutes only when score is already 20+)
if avg_min < 18:
    min_std = float(recent_min.std())
    if not pd.isna(min_std) and min_std > 10:
        return False, 0.0
```

Place this immediately after the existing `avg_min < 12` guard block (after line 220).

### Pseudocode

```
recent_min = player_df.head(10)["MIN"] as numeric
avg_min = recent_min.mean()

IF avg_min < 12: RETURN (False, 0.0)

IF avg_min < 18:
    min_std = recent_min.std()
    IF min_std > 10: RETURN (False, 0.0)   # garbage-time only

# ... rest of existing checks unchanged
```

---

## Change 3 — Tiered Star Player Prioritization

### What
Guarantee that star players (avg >= 32 min) always get at least one prop shown. After the quality gate filters props, if a star player playing today ended up with zero props, that is a data/threshold gap that should be surfaced (and corrected by relaxing the EV gate for that player).

### Two-part implementation:

**Part A — Identify today's stars before the quality gate**

In `_compute_main_page_props`, just after building `props_data` (before the quality gate at line 682), build a lookup of star players playing today:

```python
# Identify star players in today's slate (avg_min >= 32)
todays_stars: set[str] = set()
for p in props_data:
    if p.get("role") == "star":
        todays_stars.add(p["player"])
```

**Part B — Relaxed quality gate for stars**

In the quality gate loop (lines 694–704), after the standard `mp >= _MIN_MODEL_PROB and ev >= _MIN_EV and odd >= _MIN_ODDS_AMERICAN` check, add a star fallback:

```python
# Standard gate
if mp >= _MIN_MODEL_PROB and ev >= _MIN_EV and odd >= _MIN_ODDS_AMERICAN:
    quality_props.append(p)
    continue

# Star fallback: admit the best available prop for a star with no passing props yet
# Relaxed bar: model_prob >= 0.52, ev >= 0.01, odds >= -220
if (p["player"] in todays_stars
        and mp >= 0.52
        and ev >= 0.01
        and odd >= -220):
    quality_props.append(p)
```

**Part C — Tiered deduplication cap**

In the deduplication loop (lines 706–715), apply a tiered `_MAX_PER_PLAYER` based on role:

```python
_MAX_PER_PLAYER_TIER = {
    "star":     4,   # stars can have up to 4 props shown
    "starter":  3,
    "rotation": 2,
    "bench":    1,
}

for _pprops in _per_player.values():
    _pprops.sort(key=lambda x: (not x.get("is_lock", False), -(x.get("ev") or 0)))
    role_key = _pprops[0].get("role", "bench")
    cap = _MAX_PER_PLAYER_TIER.get(role_key, 1)
    deduped.extend(_pprops[:cap])
```

---

## Change 4 — Value Alt Lines at 72% of L5 Average

### What
Replace the current line-setting logic that sets lines at `floor(l5_avg) + 0.5` (approximately the average itself, ~50% hit rate) with a "value line" at 72% of the L5 average (nearest 0.5 below), giving 75-85% hit rate.

### Formula

```python
value_line = math.floor(l5_avg * 0.72 / 0.5) * 0.5
```

This floors to the nearest 0.5 below 72% of the average.

Examples:
- PTS l5_avg = 21.5 → value_line = floor(21.5 * 0.72 / 0.5) * 0.5 = floor(30.96) * 0.5 = 30 * 0.5 = 15.0
- AST l5_avg = 8.3  → value_line = floor(8.3 * 0.72 / 0.5) * 0.5 = floor(11.952) * 0.5 = 11 * 0.5 = 5.5
- REB l5_avg = 10.2 → value_line = floor(10.2 * 0.72 / 0.5) * 0.5 = floor(14.688) * 0.5 = 14 * 0.5 = 7.0

### Minimum floor per stat (existing thresholds still apply)

```python
_VALUE_LINE_MIN = {
    "PTS":  10.5,
    "AST":   3.5,
    "REB":   4.5,
    "FG3M":  1.5,
    "STL":   0.5,
    "BLK":   0.5,
}
```

### Where to change

`props_cache.py` lines 346–355 — the current line-setting block:

```python
# CURRENT (lines 346–355):
if stat_type == "PTS":
    line = math.floor(l5_avg) + 0.5 if l5_avg > 5 else 4.5
elif stat_type == "FG3M":
    line = math.floor(l5_avg) + 0.5 if l5_avg > 1 else 0.5
elif stat_type in ("STL", "BLK"):
    line = math.floor(l5_avg * 2) / 2 if l5_avg >= 0.5 else 0.5
    line = max(0.5, line)
else:  # AST, REB
    line = math.floor(l5_avg) + 0.5 if l5_avg > 2 else 1.5
```

Replace with:

```python
# NEW: value line at 72% of L5 avg (nearest 0.5 below), with per-stat minimums
_VALUE_LINE_MIN = {
    "PTS": 10.5, "AST": 3.5, "REB": 4.5,
    "FG3M": 1.5, "STL": 0.5, "BLK": 0.5,
}
raw_value_line = math.floor(l5_avg * 0.72 / 0.5) * 0.5
line = max(_VALUE_LINE_MIN.get(stat_type, 0.5), raw_value_line)
```

Note: the ML blending block (lines 362–373) runs after line is set. Keep the ML blending BUT clamp the blended result to also respect the value-line floor:

```python
# After ML blending (lines 370–371):
blended = 0.6 * ml_pred_stored + 0.4 * float(line)
raw_ml_line = math.floor(blended * 0.72 / 0.5) * 0.5
line = max(_VALUE_LINE_MIN.get(stat_type, 0.5), raw_ml_line)
```

This ensures ML predictions also produce value lines, not chalk lines.

---

## Change 5 — High-Value Stat Focus + Value Score in Quality Gate

### What
Add a `value_score` field to each prop. Props where the line is at least 60% of the player's season average get a bonus score, boosting them in the final sort. Trivial props (e.g., PTS line at 10.5 for a 28 PPG player) still pass the gate, but props where the line is >= 60% of the season average sort higher.

Also add a hard floor for PTS/AST/REB line levels to prevent sub-threshold lines from appearing:

```
PTS line must be >= 10.5 to display
AST line must be >= 3.5  to display
REB line must be >= 4.5  to display
```

### Where

**Step A — Compute value_score in `_make_prop`** (lines 455–480, add a new field):

```python
# In _make_prop return dict, add:
"value_score": round(bet_line / max(_l5_avg, 0.1), 3),
# 1.0 = line equals average (not value), 0.72 = value line, 0.60 = minimum acceptable
```

**Step B — Add meaningful-line floor filter inside quality gate** (after line 703):

```python
_MEANINGFUL_LINE_FLOORS = {"PTS": 10.5, "AST": 3.5, "REB": 4.5}

# In the quality gate loop, before appending to quality_props:
stat = p.get("stat", "")
floor_val = _MEANINGFUL_LINE_FLOORS.get(stat)
if floor_val and p.get("line", 0) < floor_val:
    continue   # skip trivially low lines
```

**Step C — Sort by value_score as secondary key** in the final sort (line 718):

```python
# CURRENT:
deduped.sort(key=lambda x: (not x.get("is_lock", False), -(x.get("ev") or 0)))

# NEW:
deduped.sort(key=lambda x: (
    not x.get("is_lock", False),
    -(x.get("ev") or 0),
    -(x.get("value_score") or 0),
))
```

---

## Change 6 — Exclude Bench Players from Alt Parlays

### What
In `build_alt_parlays` in `parlay_builder.py`, skip any alt line leg where:
- the player's `role` is `"bench"` AND
- their L5 minutes average is below 20 (not a consistent minutes getter)

Alt lines already include a `role` field via the prop dict. If the alt line dict does not have a `role` field, allow it through (defensive default).

### Where
`parlay_builder.py` — `build_alt_parlays` function, inside the `for alt in alt_lines:` loop (lines 218–249).

Add this guard immediately after the `key in used_this_parlay` check:

```python
# CURRENT structure (lines 225–231):
if key in used_this_parlay:
    continue
if not tracker.can_use(player, stat, "alt"):
    continue

# ADD after the tracker.can_use check:
# Skip garbage-time bench players (high variance, unreliable minutes)
player_role = alt.get("role", "rotation")  # default: allow if not set
if player_role == "bench":
    # Only allow bench players who have shown consistent minutes in L5
    l5_min_avg = alt.get("l5_min_avg", 0)  # this field needs to be set — see note below
    if l5_min_avg < 20:
        continue
```

### Note on `l5_min_avg` field

The alt lines dict is built in `props_cache.py` (search for where alt line dicts are constructed — around the `_ALT_WINDOWS` loop). When building alt line entries, add a `l5_min_avg` field:

```python
# When constructing an alt line entry, add:
"l5_min_avg": round(float(pd.to_numeric(player_df.head(5)["MIN"], errors="coerce").mean()), 1),
"role": role,   # already computed earlier in the player loop
```

Search `props_cache.py` for the block that appends to `alt_lines_data` (it will be near the streak computation loop) and add these two fields. The `role` variable is already in scope from line 301. Compute `l5_min_avg` from `player_df`.

---

## Change 7 — 3x 10-Leg Alt Parlays

### What
Change the `n_parlays` default and the call-site from `2` to `3`.

### Where — `parlay_builder.py`

**Function signature** (line 203):

```python
# CURRENT:
def build_alt_parlays(
    alt_lines: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 2,
    n_legs: int = 10,
) -> list[dict]:

# NEW:
def build_alt_parlays(
    alt_lines: list[dict],
    tracker: _DiversityTracker,
    n_parlays: int = 3,
    n_legs: int = 10,
) -> list[dict]:
```

**Search for the call site** in `dashboard/app.py` or wherever `build_alt_parlays` is invoked. If called with an explicit `n_parlays=2` argument, update that to `n_parlays=3`. If called without the argument (relying on the default), changing the default above is sufficient.

To find the call site:
```
grep -n "build_alt_parlays" dashboard/app.py utils/parlay_builder.py
```

---

## Implementation Order

Follow this exact order to avoid broken intermediate states:

1. **Change 2** (garbage-time exclusion in `_is_qualified_player`) — this is a pure additive guard, no dependencies.
2. **Change 1** (Overs only) — removes Under branches, simplifies `_make_prop`.
3. **Change 4** (value alt lines) — changes line calculation, depends on `_make_prop` being Over-only.
4. **Change 5** (quality gate + value score) — builds on `_make_prop` having a `value_score` field.
5. **Change 3** (star prioritization) — modifies the quality gate, depends on Changes 4 and 5 being in place.
6. **Change 6** (bench exclusion in alt parlays + `l5_min_avg` field) — add field to alt line dict first, then add guard in `parlay_builder.py`.
7. **Change 7** (n_parlays=3) — trivial constant change, last to avoid confusion.

---

## Verification Checklist

After each change, verify:

- [ ] **Change 1**: `grep "Under" utils/props_cache.py` returns zero hits in prop-emission context (only comments/blowout notes).
- [ ] **Change 2**: Paul Reed / Jamal Cain type players absent from props output on test run.
- [ ] **Change 3**: Jaylen Brown, Jayson Tatum appear in props on any BOS game day.
- [ ] **Change 4**: No PTS line above 22 for a 28 PPG player. PTS line for 21.5 L5-avg player should be ~15.0 or 15.5.
- [ ] **Change 5**: No PTS prop with line < 10.5 appears in output.
- [ ] **Change 6**: Zero bench-role players appear in alt parlays unless L5 minutes avg >= 20.
- [ ] **Change 7**: `build_alt_parlays` emits exactly 3 parlays (if >= 8 qualifying legs exist).

---

## Key Constants Summary (after all changes)

### `props_cache.py`

```python
# Line thresholds
_OVER_MIN_HIT_RATE  = 0.52        # unchanged
# _UNDER_MIN_HIT_RATE removed

# Value line formula
# raw_value_line = math.floor(l5_avg * 0.72 / 0.5) * 0.5
_VALUE_LINE_MIN = {
    "PTS": 10.5, "AST": 3.5, "REB": 4.5,
    "FG3M": 1.5, "STL": 0.5, "BLK": 0.5,
}

# Quality gate
_MIN_MODEL_PROB    = 0.54
_MIN_EV            = 0.05
_MIN_ODDS_AMERICAN = -180
_LOCK_MIN_ODDS     = -250
_MEANINGFUL_LINE_FLOORS = {"PTS": 10.5, "AST": 3.5, "REB": 4.5}

# Tiered deduplication cap
_MAX_PER_PLAYER_TIER = {
    "star": 4, "starter": 3, "rotation": 2, "bench": 1,
}

# Garbage-time guard
# avg_min < 18 AND min_std > 10  → exclude
```

### `parlay_builder.py`

```python
n_parlays: int = 3   # was 2
# bench players with l5_min_avg < 20 excluded from alt parlays
```

---

## Risk Flags

1. **Value line may overcorrect for low-average stats** — For players averaging 3.1 AST, 72% of 3.1 = 2.23, which floors to 2.0. But `_VALUE_LINE_MIN["AST"] = 3.5` catches this and bumps the line to 3.5. Confirm the minimum floor covers all edge cases.

2. **Star fallback gate may surface -220 chalk props** — The star relaxed bar allows odds down to -220. If a star's best available prop is still -230, they show nothing. This is intentional: a -230 line is a bad bet regardless of player name. Add a note in the code.

3. **`l5_min_avg` missing from existing alt line dicts** — If the field is not added before Change 6 is deployed, every bench player defaults `l5_min_avg=0` and is correctly excluded. But if there are legitimate bench breakout players (injury fill-in starters), they would also be excluded. Mitigate by ensuring the field is populated before merging Change 6.

4. **ML blending + value line interaction** — The ML blended line is also value-adjusted (Change 4). For elite scorers, ML may predict 32 PTS; 72% of 32 = 23.0 line. This is reasonable. But if ML prediction is low (8 PTS for a resting star), the value line would be 5.5, hitting the floor at 10.5. The `_VALUE_LINE_MIN` floor prevents absurd low lines in this case.
