# Plan: Fix Alt Lines Persistence + Reorganize Props Page

## Task Type
- [x] Fullstack (Backend cache logic + Frontend layout)

---

## Issue 1: 100% Alt Lines Not Persisting Through the Day

### Root Cause

`refresh_props_cache` runs every 30 minutes (scheduler). On each run it calls
`_compute_alt_lines(...)` which filters players to only those on teams playing
today. Once games start, the NBA API may return different game states (in-progress
vs. upcoming), which can cause `players_to_analyze` to shrink or the team filter
to change, resulting in fewer or zero alt_lines. The entire `_props_cache` dict
is atomically replaced, so any previously computed alt_lines are lost.

### Fix: Date-Stamped Alt Lines Cache

In `refresh_props_cache`, before computing alt_lines, check if we already have
today's alt_lines. If yes, preserve them. Only recompute when: (a) it is a new
day, or (b) the existing alt_lines list is empty.

**Changes to `utils/props_cache.py`:**

1. Add `"alt_lines_date": None` to `_props_cache` initial structure (line ~32).

2. In `refresh_props_cache`, replace the single `alt_lines = _compute_alt_lines(...)` call with:

```python
today_str = datetime.now().strftime("%Y-%m-%d")

with _cache_lock:
    existing_alt_date  = _props_cache.get("alt_lines_date")
    existing_alt_lines = _props_cache.get("alt_lines_data", [])

# Recompute only if new day OR cache was empty (first run or no results yet)
if existing_alt_date == today_str and existing_alt_lines:
    alt_lines = existing_alt_lines
    print(f"[PropsCache] Preserving {len(alt_lines)} alt lines from earlier today")
else:
    alt_lines = _compute_alt_lines(
        DF, PLAYER_POSITIONS, game_info, availability_map, players_to_analyze
    )
```

3. Add `"alt_lines_date": today_str` to the `_props_cache` dict write (inside
   the `with _cache_lock` block at the end of `refresh_props_cache`).

---

## Issue 2: Better Props Page Organization

### Current Problems
- Three separate filter rows are visually cluttered
- No at-a-glance count on each stat tab (user can't tell how many PTS props exist)
- LOCKS props are buried in the flat card grid (hidden unless user clicks LOCKS tab)
- Direction filter (Over/Under) lives on a separate row from the stat filter

### Proposed Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ [Props]  [100% Alt Lines]           ← view switcher (unchanged) │
├─────────────────────────────────────────────────────────────────┤
│ ROW 1 — Stat tabs with counts + direction inline:               │
│  [All Stats]  [PTS (12)] [AST (4)] [REB (6)] [3PM (3)]          │
│  [PRA (2)] [Combos (8)]                                          │
│  ─────────────────────────────────────────────────────          │
│  LOCKS ★ (5)  ← highlighted gold badge, always visible          │
│                                                                  │
│ ROW 2 — Secondary controls (one compact line):                  │
│  [All ↕] [Overs ↑] [Unders ↓]  │  [All] [Home] [Away]          │
│                │  Game: [All Games ▾]  Sort: [Highest EV ▾]     │
└─────────────────────────────────────────────────────────────────┘
```

Key changes:
- **Stat tabs get count badges** — show "(N)" next to each stat tab, computed
  from the filtered props data. Done client-side in the `update_props_list`
  callback by counting props per stat before rendering.
- **LOCKS become a pinned banner** — a gold-accented "LOCKS ★ (N)" button sits
  below the stat tabs, always visible. Clicking it filters to locks only.
- **Direction + Location merged into one compact row** — removes an entire row
  of visual noise.
- **No grid/card layout changes** — cards stay the same. Only filters change.

### Props List: LOCKS Pinned to Top

When viewing "All Stats" OR when the LOCKS filter is active, LOCK props always
render at the top of the list in a highlighted section before the regular props:

```
┌──────────────────────── LOCKS ★ ────────────────────────────────┐
│  [card] [card] [card]  ← green-bordered section header          │
└─────────────────────────────────────────────────────────────────┘
     then regular props below...
```

This means the `update_props_list` callback always separates `is_lock=True`
props and renders them in a distinct section, regardless of active stat filter
(unless user explicitly filters to a non-matching stat).

---

## Implementation Steps

### Step 1 — Backend: Alt Lines Persistence (`utils/props_cache.py`)
- Add `"alt_lines_date": None` to `_props_cache` init dict
- Add date-check logic in `refresh_props_cache` before `_compute_alt_lines` call
- Add `"alt_lines_date": today_str` to the cache write block

### Step 2 — Backend: Count Badges (`utils/props_cache.py`)
No changes needed — counts will be computed in the callback from existing data.

### Step 3 — Frontend: Filter Layout (`dashboard/app.py`)
In `create_best_props_page()`:
- Move direction filter tabs inline with location filter (Row 2)
- Refactor stat tabs row to include a "LOCKS ★" highlighted button
- Remove the separate direction filter row div (`id="props-filter-panel"` contents)
- Add `dcc.Store(id="props-locks-only", data=False)` for locks-only toggle

### Step 4 — Frontend: Stat Tab Count Badges (`dashboard/app.py`)
In `update_props_list` callback:
- After all filters (direction/location/game) are applied but BEFORE stat filter,
  compute per-stat counts: `{stat: count for stat in ...}`
- Pass counts as child text to stat tab components via `allow_duplicate=True` outputs
  OR compute inline and inject into a new `dcc.Store(id="props-stat-counts")`
  that the stat tab labels read from.

  **Simpler approach** (preferred): Re-render the stat tab row header from within
  `update_props_list` as a separate Output alongside `props-list.children`:
  ```
  Output("props-stat-counts-row", "children")   # new div holding count badges
  Output("props-list", "children")
  ```

### Step 5 — Frontend: LOCKS Pinned Section (`dashboard/app.py`)
In `update_props_list`:
- After filtering, split: `lock_props = [p for p in filtered if p["is_lock"]]`
- `regular_props = [p for p in filtered if not p["is_lock"]]`
- If `lock_props`:
  - Render a gold `LOCKS ★` section header + cards
  - Then render regular cards below
- If stat filter is active (non-"all") and filter is not "LOCK", only show locks
  that match the stat filter too (don't break stat filtering)

### Step 6 — CSS (`dashboard/assets/custom.css`)
- Style the count badge pill on stat tabs: small grey rounded number
- Style the LOCKS section header (gold background strip)
- Style the LOCKS banner button (gold border, distinct from regular tab)
- Compact Row 2 styling (direction + location on same line with divider)

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `utils/props_cache.py:26-37` | Modify | Add `alt_lines_date` key to init cache |
| `utils/props_cache.py:920-944` | Modify | Date-check before alt_lines recompute + store date |
| `dashboard/app.py:1592-1670` | Modify | Reorganize filter rows, add LOCKS banner, merge direction+location |
| `dashboard/app.py:2360-2430` | Modify | update_props_list: split locks, compute counts, render sections |
| `dashboard/assets/custom.css` | Modify | Badge styles, LOCKS section styles, compact row styles |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Alt lines from yesterday persisting into next day | `alt_lines_date` check resets on new `today_str` |
| Count badges causing extra callback round-trips | Compute counts inside existing `update_props_list`, no new callbacks |
| LOCKS section appearing when no locks exist | Guard with `if lock_props:` — section only renders when there are locks |
| Direction + location in one row overflowing on mobile | Use `flex-wrap: wrap` on Row 2 |
| Alt lines computed with no games (off-season) | Unchanged — `_compute_alt_lines` already handles empty `players_to_analyze` gracefully |
