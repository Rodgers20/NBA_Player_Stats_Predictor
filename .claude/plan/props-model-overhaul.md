# Implementation Plan: Props Model Overhaul + 100% Alt Lines Section

## Summary of Problems to Fix

1. **Inactive/retired players in props** — stale DF data includes players no longer active
2. **Low-minute / DNP bench players** — no minimum MPG filter, garbage-time players appear
3. **Blowout risk logic is backwards for bench players** — bench players BENEFIT from blowouts (more garbage time = OVER opportunity), but the model penalizes their overs
4. **Under props make no sense** — current threshold is barely >50% (coinflip), unders shown for bench players in blowout games
5. **Mean is skewed by outlier games** — one 50-pt explosion inflates the line; median is more robust
6. **Multi-season data polluting lines** — using all-time averages instead of current season stats
7. **No consistency scoring** — a player who always scores ~20 and one who alternates 35/5 get the same treatment
8. **Alt Lines section missing** — need a dedicated "100% ALT LINES" section showing props hit in every single recent game

---

## Part 1: Props Model Fixes

### Fix 1 — Active Player Filter (addresses problems 1 & 2)

**File:** `utils/props_cache.py`

Add a helper function `_is_qualified_player()` called early in the player loop inside `_compute_main_page_props`. Replace the current bare `len(player_df) < 5` check.

```python
def _is_qualified_player(player_name: str, player_df: pd.DataFrame) -> tuple[bool, float]:
    """
    Returns (qualified: bool, avg_min: float).
    Filters out: retired/inactive players, DNP/garbage-time players,
    players who haven't played recently.
    """
    if len(player_df) < 10:
        return False, 0.0

    # Recency: must have played within last 45 days
    most_recent = player_df["_date"].iloc[0]
    days_inactive = (datetime.now() - most_recent.to_pydatetime()).days
    if days_inactive > 45:
        return False, 0.0

    # Minimum average minutes (L10) — exclude DNP and garbage-time guys
    recent_min = pd.to_numeric(player_df.head(10)["MIN"], errors="coerce")
    avg_min = recent_min.mean()
    if pd.isna(avg_min) or avg_min < 15:   # 15 MPG minimum
        return False, 0.0

    # Current season: must have at least 10 games this season
    if "SEASON" in player_df.columns:
        current_season_rows = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
        if len(current_season_rows) < 10:
            return False, 0.0

    return True, float(avg_min)
```

**Usage in player loop (replaces bare `< 5` check):**
```python
qualified, avg_min = _is_qualified_player(player_name, player_df)
if not qualified:
    continue
```

---

### Fix 2 — Role Classification for Blowout Logic (addresses problem 3)

**File:** `utils/props_cache.py`

Add a role helper and completely rewrite the blowout EV penalty.

```python
def _get_player_role(avg_min: float) -> str:
    """Classify player role by average minutes."""
    if avg_min >= 30:   return "star"       # Franchise player, always starts
    elif avg_min >= 24: return "starter"    # Regular starter
    elif avg_min >= 17: return "rotation"   # Key rotation / bench starter
    else:               return "bench"      # Reserve / end-of-bench
```

**Replace current blowout block (uniform penalty) with role-aware logic:**

```python
# ── Blowout-risk EV adjustment (role-aware) ───────────────────────────────
if blowout_risk:
    role = _get_player_role(avg_min)
    spread_factor = 0.25 if blowout_spread >= 15 else 0.12

    if role in ("star", "starter"):
        # Stars get RESTED in blowouts → fewer minutes → OVERs suffer
        if direction == "Over":
            ev_value *= (1 - spread_factor)
        # UNDERs for starters in blowouts become more attractive
        elif direction == "Under":
            ev_value *= (1 + spread_factor * 0.5)  # Modest boost

    elif role in ("rotation", "bench"):
        # Bench/rotation get GARBAGE TIME → more minutes → OVERs improve
        if direction == "Over":
            ev_value *= (1 + spread_factor * 0.4)   # Modest boost
        # UNDERs for bench in blowout make NO sense — tank them
        elif direction == "Under":
            ev_value *= 0.30   # Severely reduce, will fail the EV threshold

# is_lock: only lock starters' overs when no blowout
is_lock = (
    hit_rate >= 0.80 and n >= 5
    and not (blowout_risk and role in ("star", "starter") and direction == "Over")
)
```

---

### Fix 3 — Raise Under Threshold & Add Contextual Guard (addresses problem 4)

**File:** `utils/props_cache.py`

```python
OVER_MIN_HIT_RATE  = 0.52   # Need >52% (was >50%) — avoid pure coinflips
UNDER_MIN_HIT_RATE = 0.62   # Need >62% for unders — much higher bar

# Over
if hit_rate_over > OVER_MIN_HIT_RATE:
    props_data.append(_make_prop("Over", over_line, hit_rate_over, hits_over))

# Under — additional context guard
under_makes_sense = True
if blowout_risk and role in ("rotation", "bench"):
    under_makes_sense = False   # Bench gets more time in blowouts, not less

if (hit_rate_under > UNDER_MIN_HIT_RATE
        and hit_rate_under != hit_rate_over
        and under_makes_sense):
    props_data.append(_make_prop("Under", under_line, hit_rate_under, hits_under))
```

---

### Fix 4 — Median-Based Lines (addresses problem 5)

**File:** `utils/props_cache.py`

Replace mean-based line calculation with median-based:

```python
import math

avg_stat   = recent_stats.mean()     # Keep for display/insight
median_stat = recent_stats.median()  # Use for line setting (robust to outliers)

if avg_stat < 1:
    continue

# Line from MEDIAN (more robust to single blowup games)
if stat_type == "PTS":
    line = math.floor(median_stat) + 0.5 if median_stat > 5 else 4.5
elif stat_type == "FG3M":
    line = math.floor(median_stat) + 0.5 if median_stat > 1 else 0.5
else:  # AST, REB
    line = math.floor(median_stat) + 0.5 if median_stat > 2 else 1.5
```

Note: `avg_stat` is still stored in the prop dict for display ("averages X.X per game").

---

### Fix 5 — Current Season Priority (addresses problem 6)

**File:** `utils/props_cache.py`

Replace `recent_10 = player_df.head(10)` with current-season-first logic:

```python
# Prefer current season data for all statistical calculations
if "SEASON" in player_df.columns:
    current_df = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
    if len(current_df) >= 10:
        recent_10 = current_df.head(10)
    elif len(current_df) >= 5:
        recent_10 = current_df.head(len(current_df))   # Use what we have
    else:
        recent_10 = player_df.head(10)   # Fallback: cross-season
else:
    recent_10 = player_df.head(10)

# Also scope home/away splits to current season
if "SEASON" in player_df.columns:
    season_df = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
else:
    season_df = player_df

home_games = season_df[season_df["MATCHUP"].str.contains("vs.", na=False)].head(10)
away_games = season_df[season_df["MATCHUP"].str.contains("@",   na=False)].head(10)
```

---

### Fix 6 — Consistency Score (addresses problem 7)

**File:** `utils/props_cache.py`

Add consistency multiplier to EV calculation inside `_make_prop`:

```python
# Inside _make_prop closure (after ev_value is calculated):
std_stat = recent_stats.std()
cv = std_stat / avg_stat if avg_stat > 0 else 1.0   # Coefficient of variation
# CV interpretation:
#   < 0.20 = very consistent (low variance)  → full EV
#   0.20–0.40 = moderately consistent        → slight reduction
#   > 0.40 = volatile / unpredictable        → significant reduction
consistency_mult = max(0.75, 1.0 - max(0.0, cv - 0.20) * 0.60)
ev_value *= consistency_mult
```

Store in prop dict: `"consistency": round(1 - cv, 2)` for display in UI.

---

## Part 2: 100% Alt Lines Section

### Step 1 — Add `_compute_alt_lines()` to props_cache.py

**File:** `utils/props_cache.py`

```python
# ── ALT LINE CONSTANTS ────────────────────────────────────────────────────
_ALT_LINE_WINDOWS   = [5, 6, 7, 8, 10, 12, 15, 17, 18, 20]
_ALT_MIN_THRESHOLD  = {"PTS": 5, "AST": 2, "REB": 3, "FG3M": 1}
_ALT_STAT_LABELS    = {"PTS": "POINTS", "AST": "ASSISTS",
                       "REB": "REBOUNDS", "FG3M": "MADE THREES"}


def _compute_alt_lines(DF, PLAYER_POSITIONS, game_info) -> list[dict]:
    """
    Find the highest threshold a player hit in 100% of their last N games.

    Example: LaMelo Ball scores 15+ Points in every single one of his
    last 18 games → alt line = "15+ POINTS, 18/L18"

    This surfaces alt lines (below the standard over/under line) that have
    genuinely hit every game — ideal for parlay legs and high-confidence plays.
    """
    if not game_info["has_todays_games"] or PLAYER_POSITIONS.empty:
        return []

    players_today = PLAYER_POSITIONS[
        PLAYER_POSITIONS["TEAM_ABBREVIATION"].isin(game_info["teams_playing"])
    ]["PLAYER_NAME"].tolist()

    alt_lines = []

    for player_name in players_today:
        player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False)

        if len(player_df) < 5:
            continue

        # Recency check
        if (datetime.now() - player_df["_date"].iloc[0].to_pydatetime()).days > 30:
            continue

        # Minimum minutes
        avg_min = pd.to_numeric(player_df.head(10)["MIN"], errors="coerce").mean()
        if pd.isna(avg_min) or avg_min < 15:
            continue

        # Use current season data preferentially
        if "SEASON" in player_df.columns:
            season_df = player_df[player_df["SEASON"].str.startswith("2025", na=False)]
            if len(season_df) >= 5:
                stat_df = season_df
            else:
                stat_df = player_df
        else:
            stat_df = player_df

        player_team = _get_player_team(player_name, PLAYER_POSITIONS)
        opponent    = _resolve_opponent(player_name, player_team, player_df, game_info)

        for stat_type in ["PTS", "AST", "REB", "FG3M"]:
            if stat_type not in stat_df.columns:
                continue

            stat_series = pd.to_numeric(stat_df[stat_type], errors="coerce")
            min_threshold = _ALT_MIN_THRESHOLD.get(stat_type, 1)

            best_alt = None  # (threshold, window, score)

            for n in _ALT_LINE_WINDOWS:
                if len(stat_series) < n:
                    continue

                last_n = stat_series.iloc[:n]
                if last_n.isna().any():
                    continue

                # The highest threshold hit 100% of the time = floor of minimum
                floor_thresh = int(last_n.min())
                if floor_thresh < min_threshold:
                    continue

                if not (last_n >= floor_thresh).all():
                    continue   # Shouldn't happen but safety check

                # Score: longer window × how high threshold is relative to mean
                mean_val = last_n.mean()
                score = n * (floor_thresh / mean_val) if mean_val > 0 else 0

                if best_alt is None or score > best_alt[2]:
                    best_alt = (floor_thresh, n, score)

            if best_alt is None:
                continue

            threshold, window, score = best_alt

            # Only surface meaningful streaks (at least L5, meaningful threshold)
            if window < 5:
                continue

            alt_lines.append({
                "player":     player_name,
                "team":       player_team,
                "opponent":   opponent,
                "stat":       stat_type,
                "stat_label": _ALT_STAT_LABELS.get(stat_type, stat_type),
                "threshold":  threshold,
                "window":     window,
                "trend":      f"{window}/L{window}",
                "prop_label": f"{threshold}+ {_ALT_STAT_LABELS.get(stat_type, stat_type)}",
                "score":      round(score, 2),
                "avg_stat":   round(
                    pd.to_numeric(stat_df.head(window)[stat_type], errors="coerce").mean(), 1
                ),
            })

    # Sort by window DESC (longest first), then by score DESC
    alt_lines.sort(key=lambda x: (-x["window"], -x["score"]))

    # De-duplicate: keep the single best alt line per (player, stat) combo
    seen, deduped = set(), []
    for alt in alt_lines:
        key = f"{alt['player']}|{alt['stat']}"
        if key not in seen:
            seen.add(key)
            deduped.append(alt)

    return deduped[:25]   # Top 25 alt lines
```

---

### Step 2 — Integrate Alt Lines into Cache

**File:** `utils/props_cache.py`, function `refresh_props_cache()`

```python
# Add to _props_cache initial state (line ~27):
_props_cache = {
    ...existing keys...
    "alt_lines_data": [],    # NEW
}

# Add to refresh_props_cache(), after main_data computation:
alt_lines = _compute_alt_lines(DF, PLAYER_POSITIONS, game_info)

# Add to _props_cache write block:
_props_cache = {
    "main_page_data": main_data,
    "callback_data":  callback_data,
    "sidebar_data":   sidebar_data,
    "alt_lines_data": alt_lines,     # NEW
    ...existing keys...
}
```

---

### Step 3 — Create Alt Lines UI Function

**File:** `dashboard/app.py`

Add `create_alt_lines_section(alt_lines)` function before `create_best_props_page()`.

**Design spec (matching SmartPicks reference images):**

```
┌──────────────────────────────────────────────────────┐
│  🔥 100% ALT LINES                                   │  ← dark header
│  Players who hit these thresholds in every game      │
├─────┬──────────────────────┬──────────────┬──────────┤
│ TM  │ PLAYER               │ PROP         │ TREND    │  ← column headers
├─────┼──────────────────────┼──────────────┼──────────┤
│ LAL │ LeBron James         │ 15+ POINTS   │ 10/L10   │  ← green row
│ BOS │ Jayson Tatum         │ 2+ THREES    │ 8/L8     │
│ DEN │ Nikola Jokic         │ 8+ REBOUNDS  │ 7/L7     │
...
└──────────────────────────────────────────────────────┘
```

**Color scheme:**
- Section background: `#060b18` (same as app bg)
- Row background alternating: `rgba(34,197,94,0.10)` / `rgba(34,197,94,0.06)`
- PROP column: `#22c55e` text on dark bg
- TREND column: `#22c55e` text, bold
- Team logo: 28×28px rounded square
- Header: white/teal text, uppercase

**Key HTML structure (pseudo):**
```python
html.Div([
    # Section header
    html.Div([
        html.Span("🔥 100% ALT LINES", className="alt-lines-title"),
        html.Span(f"{len(alt_lines)} streaks today", className="alt-lines-subtitle"),
    ], className="alt-lines-header"),

    # Table
    html.Div([
        # Column headers
        html.Div(["TEAM", "PLAYER", "PROP", "TREND"], className="alt-lines-col-headers"),
        # Rows
        *[create_alt_line_row(alt) for alt in alt_lines]
    ], className="alt-lines-table"),

], id="alt-lines-section", className="alt-lines-section")
```

Each row:
```python
def create_alt_line_row(alt):
    team_logo = get_team_logo_url(alt["team"])
    return html.Div([
        # Team logo
        html.Img(src=team_logo, className="alt-row-logo"),
        # Player name
        html.Span(alt["player"], className="alt-row-player"),
        # Prop (green highlight)
        html.Span(alt["prop_label"], className="alt-row-prop"),
        # Trend
        html.Span(alt["trend"], className="alt-row-trend"),
    ], className="alt-line-row")
```

---

### Step 4 — CSS for Alt Lines Section

**File:** `dashboard/assets/custom.css`

```css
/* ── 100% Alt Lines Section ─────────────────────────────────────────────── */
.alt-lines-section {
    background: var(--bg-secondary);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 32px;
}

.alt-lines-header {
    padding: 20px 24px 16px;
    border-bottom: 1px solid rgba(34, 197, 94, 0.15);
    display: flex;
    align-items: baseline;
    gap: 12px;
}

.alt-lines-title {
    font-size: 1.4rem;
    font-weight: 900;
    color: #22c55e;
    letter-spacing: -0.02em;
    text-transform: uppercase;
}

.alt-lines-subtitle {
    font-size: 0.8rem;
    color: var(--text-muted);
}

.alt-lines-col-headers {
    display: grid;
    grid-template-columns: 48px 1fr 180px 100px;
    padding: 8px 16px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-subtle);
}

.alt-line-row {
    display: grid;
    grid-template-columns: 48px 1fr 180px 100px;
    padding: 12px 16px;
    align-items: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    transition: background 0.15s;
}

.alt-line-row:hover {
    background: rgba(34, 197, 94, 0.08);
}

.alt-row-logo {
    width: 28px;
    height: 28px;
    object-fit: contain;
    border-radius: 6px;
}

.alt-row-player {
    font-size: 0.9rem;
    font-weight: 700;
    color: #f0f4ff;
    padding-left: 8px;
}

.alt-row-prop {
    font-size: 0.875rem;
    font-weight: 800;
    color: #22c55e;
    background: rgba(34, 197, 94, 0.12);
    padding: 4px 10px;
    border-radius: 6px;
    text-align: center;
    letter-spacing: 0.02em;
}

.alt-row-trend {
    font-size: 0.875rem;
    font-weight: 800;
    color: #22c55e;
    text-align: center;
}
```

---

### Step 5 — Wire Alt Lines into `create_best_props_page()`

**File:** `dashboard/app.py`, inside `create_best_props_page()`

```python
cache         = get_cached_props()
props_data    = cache["main_page_data"]
alt_lines     = cache.get("alt_lines_data", [])   # NEW
...

return html.Div([
    html.Div([
        # LEFT: player panel
        html.Div(id="props-player-panel", ...),

        # RIGHT: props content
        html.Div([
            # Page header (existing)
            ...

            # ── NEW: 100% Alt Lines section ─────────────────────────
            create_alt_lines_section(alt_lines) if alt_lines else None,

            # ── Existing: stat type filter tabs ──────────────────────
            # Row 1: Stat type filter
            ...
```

---

## Key Files Modified

| File | Change | Description |
|------|--------|-------------|
| `utils/props_cache.py` | Modify | Add `_is_qualified_player()`, `_get_player_role()`, `_compute_alt_lines()` |
| `utils/props_cache.py` | Modify | Fix blowout logic, raise under threshold, median lines, consistency score |
| `utils/props_cache.py` | Modify | Add `alt_lines_data` to cache dict |
| `dashboard/app.py` | Modify | Add `create_alt_lines_section()`, `create_alt_line_row()` functions |
| `dashboard/app.py` | Modify | Wire alt lines into `create_best_props_page()` |
| `dashboard/app.py` | Modify | Add `props-reload-interval` to also refresh alt lines store (or re-use page refresh) |
| `dashboard/assets/custom.css` | Modify | Add `.alt-lines-*` CSS classes |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `_is_qualified_player` too strict — filters real players | Start with 15 MPG / 10-game thresholds; tune downward if needed |
| `_compute_alt_lines` slow — iterates all players × stats × windows | O(players × 4 × 10) = ~2,400 iterations max, all in-memory; fast |
| Alt lines empty on first load | Already handled by `props-reload-interval` refresh mechanism |
| Role misclassification edge cases (load management, injury returns) | Use L10 average minutes; injury-returning players filtered by `_is_qualified_player` recency check |
| Median line differs significantly from Vegas line | When live odds available, Vegas line always overrides — this only affects synthetic lines |
| Under threshold 62% may produce zero under props | Acceptable — unders are low-value bets; if truly a strong under, it will clear 62% |

---

## Implementation Order

1. `_is_qualified_player()` + `_get_player_role()` helpers
2. Fix blowout role logic in `_compute_main_page_props`
3. Raise under threshold + contextual guard
4. Median-based line calculation
5. Current season priority filter
6. Consistency score multiplier
7. `_compute_alt_lines()` function
8. Cache integration (add `alt_lines_data`)
9. `create_alt_lines_section()` + CSS
10. Wire alt lines into Best Props page

### SESSION_ID
- CODEX_SESSION: N/A (ace-tool not available — Claude Code native analysis used)
- GEMINI_SESSION: N/A
