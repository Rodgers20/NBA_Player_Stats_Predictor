# Implementation Plan: Props Projection Overhaul v2

## Summary of Issues
1. **Line too high vs L5 avg** — projection pulls line above what player realistically hits (e.g. Booker 31.5 avg → model shows 33)
2. **Chart window switching broken** — clicking L10/L20/Home/Away tabs does nothing
3. **Combo stat filters show nothing** — clicking PTS+AST, PTS+REB, REB+AST, PRA, DD, TD shows blank
4. **Sorting wrong** — 60% conf card appears below 20% conf cards
5. **Games fallback** — already works in data_fetch.py but props page shows "no games" message
6. **Parlay counts** — need exactly: 5 over, up to 5 ML (fewer if not enough), 1 totals, 3 alt
7. **Alt lines stale** — some alt line thresholds are from old data

---

## Task Type
- [x] Fullstack (Backend + Frontend)

---

## Implementation Steps

### Step 1 — Fix line/projection formula (`utils/props_cache.py:509`)

**Root cause**: `raw_line = math.floor(proj * 0.92 / 0.5) * 0.5` uses the *projection* (which can be boosted by ML, defense, home/away factors). When ML+context inflates `proj` above `l5_avg`, the line ends up higher than the player's real recent output.

**Fix**: Set the line at the **20th percentile of L5 values** — meaning 80%+ of L5 games beat the line. Fall back to `floor(l5_avg * 0.82 / 0.5) * 0.5` when fewer than 5 data points exist.

```python
# In props_cache.py, replace lines 509-510:

# Line = 20th-percentile of L5 values → ≥80% of last 5 games hit it
if len(_l5_source) >= 3:
    _p20 = float(pd.to_numeric(pd.Series(list(_l5_source.values[:5])),
                                errors="coerce").quantile(0.20))
    raw_line = math.floor(max(_p20, l5_avg * 0.75) / 0.5) * 0.5
else:
    raw_line = math.floor(l5_avg * 0.82 / 0.5) * 0.5

line = max(_VALUE_LINE_MIN.get(stat_type, 0.5), raw_line)
```

**Also fix sim_book_line** (line 515) — books set at ~l5_avg, not above:
```python
# Remove the extra buffer that pushes line above l5_avg
sim_book_line = math.floor(l5_avg / 0.5) * 0.5
```

**Effect**: Booker L5 avg 31.5 → line ~25.5–26.5 (hit 80%+). Deni avg 26.5 → line ~22.5–23.5.

---

### Step 2 — Fix chart window switching (`dashboard/app.py`)

**Root cause**: The `render_prop_chart_window` MATCH callback fires when the `prop-chart-window` store changes, but `update_prop_window_store` (ALL) only updates the correct store when `out_id["id"]["index"] == card_idx`. If the card_idx has special characters or the index mismatch is subtle, the store never updates.

**Also**: For Home/Away windows, if `home_games` is empty, `chart_windows["home"]` has empty values → chart renders blank → user sees no change.

**Fixes**:

1. In `render_prop_chart_window`: Add fallback to L5 when requested window is empty:
```python
def render_prop_chart_window(window, chart_data):
    win_data = chart_data.get(window) or {}
    if not win_data.get("values"):
        win_data = chart_data.get("l5") or {"values": [], "labels": []}
    ...
```

2. In `update_prop_window_store`: Print debug to verify card_idx matching, and simplify the match logic:
```python
# Instead of comparing string index, use split logic more robustly
card_idx = triggered_index.rsplit("|", 1)[0]  # already correct
# Verify: store index format is "PlayerName|STAT" (no window suffix)
# Tab index format is "PlayerName|STAT|l10" → split gives card_idx = "PlayerName|STAT"
```

3. Ensure `_extract_chart_window` always returns at least 1 data point (prevents empty window silently breaking):
```python
def _extract_chart_window(df, stat_type):
    if df is None or df.empty:
        return [], []
    ...
```

---

### Step 3 — Fix combo stat filters (`dashboard/app.py`)

**Root cause**: The filter buttons for PTS+AST, PTS+REB etc. set `stat_filter = "PTS+AST"`, and the filter is `p.get("stat") == stat_filter`. The combo stat is stored as `"+".join(combo_stats)` which gives `"PTS+AST"`. **This should match** — but DD/TD props store stat as `"DD"` / `"TD"` correctly too.

**Real problem**: The stat filter button IDs (`props-filter-pa`, `props-filter-pr`, etc.) exist in the callback but need to be verified they exist in the layout HTML. If the layout is missing those button elements, no click ever fires.

**Fix**: Check layout and ensure all 12 filter buttons are rendered with correct IDs. Read `app.py` around line 2440 to verify the filter tab layout contains:
- `props-filter-pa` (PTS+AST)
- `props-filter-pr` (PTS+REB)  
- `props-filter-ar` (AST+REB)
- `props-filter-pra` (PRA)
- `props-filter-dd` (DD)
- `props-filter-td` (TD)

If any are missing, add them. Also ensure the stat count badges show counts for combo stats so user sees how many props exist per category.

---

### Step 4 — Fix sorting (already partially done, verify)

The sort by `hit_rate_l5` was added in the previous session. Verify it is working by checking `update_props_list` callback uses `hit_rate_l5` as primary sort key. The confidence % shown on each card should match `hit_rate_l5` so cards with 80%+ appear at top.

**Also**: The confidence % displayed (`conf_pct`) uses `hit_rate_vs_book or hit_rate` — after fixing the line to be 20th-percentile based, most props will have `hit_rate_l5 ≥ 0.80`, so the conf% will correctly show 80%+.

---

### Step 5 — Parlay count fix (`utils/parlay_builder.py`)

**User wants**:
- Exactly **5 OVER parlays** (hard target)
- Up to **5 ML parlays** (build as many as games allow, cap at 5)
- Exactly **1 Game Totals parlay**
- Exactly **3 Alt Lines parlays** (100% hit rate)

**Changes to `build_all_parlays`**:
```python
# Step A: Hard-code over parlay count to 5
all_overs = build_over_parlays(props, tracker, n_parlays=5, n_legs=3)

# Step B: ML parlays — build up to 5 (build_ml_trio already handles 3; replace with build_ml_parlays(n=5))
ml_parlays = build_ml_parlays(game_predictions, tracker, n_parlays=5)

# Step C: Totals — build exactly 1
total_parlays = build_totals_parlays(game_predictions, tracker, n_parlays=1)

# Step D: Alt lines — 3 parlays
alt_parlays = build_alt_parlays(alt_lines, tracker, n_parlays=3, n_legs=10)
```

**`build_ml_parlays` new function** (replaces `build_ml_trio`):
```python
def build_ml_parlays(game_predictions, tracker, n_parlays=5):
    """Build up to n_parlays 3-leg ML parlays. Each uses a different combination of games."""
    # Group picks by confidence, build combos of 3
    # If fewer than 3 HIGH/MEDIUM picks exist, build 2-leg or skip
    # Return however many we can build, up to n_parlays
```

---

### Step 6 — Fix alt lines staleness (`utils/props_cache.py`)

**Root cause**: Alt lines use `player_df.head(N)` which is sorted by `_date` desc. If player has recent inactive games (DNP), those get included in the streak window and break the streak count.

**Fix** in `_compute_alt_lines`: Filter out games where `MIN < 10` (DNP/garbage time) before computing streaks, so the threshold is computed from real playing games only:

```python
# In _compute_alt_lines, before streak computation:
_active_df = player_df[pd.to_numeric(player_df["MIN"], errors="coerce") >= 10]
# Use _active_df instead of player_df for streak windows
```

---

## Key Files

| File | Line Range | Operation | Description |
|------|-----------|-----------|-------------|
| `utils/props_cache.py` | L509-515 | Modify | Fix line formula to 20th-percentile of L5 |
| `utils/props_cache.py` | L1300-1350 | Modify | Filter DNP games before alt line streak |
| `utils/parlay_builder.py` | L1100-1160 | Modify | Hard-code 5 over, ≤5 ML, 1 totals, 3 alt |
| `utils/parlay_builder.py` | L134-193 | Modify | Replace `build_ml_parlay` → `build_ml_parlays(n=5)` |
| `dashboard/app.py` | L3625-3635 | Modify | Chart window fallback when window is empty |
| `dashboard/app.py` | L2440-2510 | Verify/Fix | Ensure all 12 filter tabs exist in layout |
| `dashboard/app.py` | L3066-3090 | Verify | Stat filter mapping includes all combo IDs |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| 20th-percentile gives line too low for inconsistent players | Floor at `l5_avg * 0.70` minimum |
| ML parlays can't be built (< 3 games) | Return however many available (0–5), don't error |
| Combo filter still empty after fix | Add console log in callback to verify stat values match filter |
| Alt line staleness persists | Also add `_date` recency check — only include alts from last 30 days |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (no external models used)
- GEMINI_SESSION: N/A
