---
# Implementation Plan: Game Predictor & Dashboard Improvements

## Overview
9 grouped features across 4 areas: game predictor intelligence, prediction tracking/Excel export, player analysis fixes, and best props enhancements.

---

## AREA 1 — Game Predictor Intelligence

### 1A. Injury-Aware Score Projections

**Problem:** `predict_game()` ignores injured players entirely. If Cade Cunningham (WAS's top scorer, ~25 PPG) is OUT, the model still predicts as if he's playing.

**Solution:** Before blending form + season ratings, deduct missing key scorers' average contribution from the team's projected score.

**File:** `utils/game_predictor.py`

**New method:** `_apply_injury_penalty(team, raw_score, player_logs_df, injury_status_fn) -> (adjusted_score, list[str])`

**Logic:**
```python
def _apply_injury_penalty(self, team, raw_score, injuries):
    # injuries = list of {"name": str, "status": str, "reason": str}
    penalty = 0.0
    missing_notes = []
    for player in injuries:
        if player["status"] in ("OUT", "DOUBTFUL"):
            # Look up player's L10 PPG from self.logs
            player_df = self.logs[self.logs["PLAYER_NAME"] == player["name"]]
            if len(player_df) >= 3:
                ppg = player_df.sort_values("_date", ascending=False).head(10)["PTS"].mean()
                weight = 0.7 if player["status"] == "OUT" else 0.3
                penalty += ppg * weight
                missing_notes.append((player["name"], ppg, player["status"]))
    adjusted = max(raw_score - penalty, raw_score * 0.75)  # floor at 75% of original
    return round(adjusted, 1), missing_notes
```

**Integration:** `predict_game()` gains optional param:
```python
def predict_game(self, home: str, away: str,
                 home_injuries: list[dict] | None = None,
                 away_injuries: list[dict] | None = None) -> dict:
```

After computing `pred_home`/`pred_away` but before pace adjustment:
```python
if home_injuries:
    pred_home, home_missing = self._apply_injury_penalty(home, pred_home, home_injuries)
if away_injuries:
    pred_away, away_missing = self._apply_injury_penalty(away, pred_away, away_injuries)
```

Store `home_missing` and `away_missing` in return dict for use by `_build_intel()` and reasoning builder.

**Caller update** (`dashboard/app.py` → `create_todays_games_page()`):
```python
from utils.injury_news import get_team_injuries
home_injuries = get_team_injuries(home_team_internal)
away_injuries = get_team_injuries(away_team_internal)
prediction = GAME_PREDICTOR.predict_game(
    home_team_internal, away_team_internal,
    home_injuries=home_injuries,
    away_injuries=away_injuries,
)
```

---

### 1B. Pick Reasoning / Natural-Language Explanations

**Problem:** Model picks have no explanations. User can't tell if a pick is data-driven or random.

**Solution:** New function `build_pick_reasoning()` that generates per-pick explanations.

**File:** `utils/game_predictor.py` — new method `build_pick_reasoning()`

**Return shape:**
```python
{
  "winner_reason": str,   # Why this team wins
  "spread_reason": str,   # Why this spread pick
  "total_reason": str,    # Why OVER or UNDER
  "key_factors": list[str],  # 3-5 bullet points shown in UI
}
```

**Logic per pick type:**

*Winner reasoning:*
```
"{winner} projected {pred_winner:.0f}-{pred_loser:.0f}.
Form: {winner} {w}-{l} L10 ({rolling_ppg:.0f} PPG) vs {loser} {w2}-{l2} L10.
[If injuries]: {loser} missing {player} ({ppg:.0f} PPG, {status}).
[If H2H dominant]: {winner} leads H2H {wins}-{losses} in last {total} meetings.
[If home]: Home court advantage (+2.5 pts)."
```

*Total reasoning:*
```
"Model projects {pred_total:.0f} pts total ({home} {pred_home:.0f} + {away} {pred_away:.0f}).
[If OVER]: {home} avg {home_ppg:.0f} PPG L10, {away} avg {away_ppg:.0f} PPG L10.
Both teams pace up (avg pace {pace:.0f} vs league {league_pace:.0f}).
Model line {pred_total:.0f} vs book line {actual_total:.0f} → edge of {edge:.1f} pts."
[If UNDER]: "{team} elite D (#{rank} in pts allowed, {opp_ppg:.0f} PPG allowed)"
```

*Spread reasoning:*
```
"Model has {team} by {model_spread:.1f}, book has {team} by {book_spread:.1f}.
Edge of {edge:.1f} pts → {pick} {spread_confidence} confidence.
[Key factor bullets same as winner]"
```

**UI Integration:** In `create_todays_games_page()`, add a "WHY" expandable section below each odds col or as a tooltip/modal trigger. Replace the existing `MATCHUP INTEL` section (ROW 5) with richer `pick_reasoning["key_factors"]` bullets.

**Display format per odds column:**
- Under each MODEL PICK badge, add small italic text: the 1-line reason
- Example under OVER badge: *"DET 118 + WAS 117 PPG avg → 235 proj, book at 222.5"*

---

### 1C. Game Start Time Fix

**Problem:** Top-right shows `datetime.now().strftime("%-I:%M %p ET")` — always shows current time.

**File:** `dashboard/app.py` — `create_todays_games_page()`, line ~1298

**Fix:** ESPN API already returns game time. In `get_todays_games()` result, check for columns `GAME_STATUS_TEXT`, `GAME_TIME`, or `HOME_TEAM_TIME`.

```python
# In create_todays_games_page() game loop:
game_time_str = game.get("GAME_TIME", game.get("GAME_STATUS_TEXT", ""))
# Parse and display as game start time, not current time
# If it's a timestamp string like "2026-03-19T00:00:00Z", format it
# If it looks like "7:30 pm ET", display as-is
# Fall back to "TBD" if missing
```

**In the card header (ROW 1):**
```python
# Replace:
datetime.now().strftime("%-I:%M %p ET") if not is_live else ""
# With:
_format_game_time(game_time_str) if not is_live else ""
```

Add helper:
```python
def _format_game_time(raw_time: str) -> str:
    """Parse game time from ESPN API into display format."""
    if not raw_time or raw_time in ("", "—"):
        return "TBD"
    # Try ISO parse
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        # Convert UTC to ET
        et_offset = timedelta(hours=-4)  # EDT
        dt_et = dt + et_offset
        return dt_et.strftime("%-I:%M %p ET")
    except:
        return raw_time  # Return raw if already formatted
```

---

### 1D. Injury Panel → Collapsible Dropdown

**Problem:** Injury panel is always expanded, taking up space even when teams are fully healthy.

**Solution:** Replace always-expanded panel with a collapsible `<details>`-style toggle button. Show count badge ("3 injuries") collapsed, expand on click.

**File:** `dashboard/app.py` — `_injury_panel()` function (Lines 981-1046)

**New Structure:**
```python
def _injury_panel(home_team, away_team):
    home_inj = get_team_injuries(home_team)
    away_inj = get_team_injuries(away_team)
    all_injuries = home_inj + away_inj

    if not all_injuries:
        return None  # Don't show panel if no injuries

    out_count = sum(1 for p in all_injuries if p["status"] == "OUT")
    total_count = len(all_injuries)

    # Build collapsed toggle header
    badge_color = "#f43f5e" if out_count > 0 else "#f59e0b"
    toggle_label = f"{out_count} OUT" if out_count else f"{total_count} Questionable"

    # Use dcc.Store + callback for toggle, OR use HTML details/summary element
    # Simplest: use html.Details + html.Summary (native HTML, no callback needed)
    return html.Details([
        html.Summary([
            html.Span("INJURY REPORT", style={...label style...}),
            html.Span(toggle_label, style={"background": badge_color, ...badge style...}),
        ], style={"cursor": "pointer", "display": "flex", "alignItems": "center", "gap": "8px"}),
        html.Div([
            # existing injury chips for home + away
            _render_team_injuries(home_team, home_inj),
            _render_team_injuries(away_team, away_inj),
        ], style={"marginTop": "10px"})
    ], style={...panel container style...})
```

---

## AREA 2 — Prediction Tracking & Excel Export

### 2A. New File: `utils/prediction_tracker.py`

**Purpose:** Store each day's predictions before games start. After games finish, grade them. Export to Excel.

**Storage:** JSON file at `data/prediction_history.json`

**Schema:**
```json
{
  "2026-03-19": {
    "generated_at": "2026-03-19T12:00:00",
    "graded_at": "2026-03-19T23:30:00",
    "games": [
      {
        "game_id": "DET@WAS",
        "home": "WAS", "away": "DET",
        "game_time": "7:30 PM ET",
        "predictions": {
          "winner_pick": "DET",
          "winner_confidence": "MEDIUM",
          "spread_pick": "AWAY",
          "spread_team": "DET",
          "spread_line": -3.5,
          "spread_confidence": "MEDIUM",
          "total_pick": "OVER",
          "total_line": 222.5,
          "total_confidence": "HIGH",
          "model_home_score": 114.2,
          "model_away_score": 120.8,
          "model_total": 235.0,
          "reasoning": {...}
        },
        "actuals": {
          "home_score": 109,
          "away_score": 118,
          "winner": "DET",
          "total": 227
        },
        "grades": {
          "winner_correct": true,
          "spread_correct": true,
          "total_correct": true
        }
      }
    ],
    "summary": {
      "moneyline": {"correct": 3, "total": 5, "pct": 60.0},
      "spread":    {"correct": 2, "total": 5, "pct": 40.0},
      "total":     {"correct": 4, "total": 5, "pct": 80.0}
    }
  }
}
```

**Key functions:**
```python
def save_daily_predictions(date: str, games: list[dict]) -> None:
    """Save model predictions for the day. Called after predictions are built."""

def grade_predictions(date: str) -> dict:
    """
    Fetch actual game results from ESPN API and grade stored predictions.
    Called by scheduler at ~midnight after games finish.
    Returns graded summary dict.
    """

def export_to_excel(output_path: str = "data/prediction_report.xlsx") -> str:
    """
    Export full prediction history to Excel with 4 sheets:
    - Summary: Date, ML%, Spread%, O/U%
    - Moneyline: Per-game ML picks + results
    - Spread: Per-game spread picks + results
    - Total: Per-game O/U picks + results
    Returns path to written file.
    """
```

**Excel structure (using openpyxl):**
- Sheet 1 "Summary": Date | ML Correct | ML Total | ML% | Spread% | O/U% | Overall%
- Sheet 2 "Moneyline": Date | Game | Pick | Confidence | Result | ✓/✗
- Sheet 3 "Spread": Date | Game | Pick | Line | Model Line | Edge | Result | ✓/✗
- Sheet 4 "Total": Date | Game | Pick | Line | Model Total | Edge | Result | ✓/✗
- Conditional formatting: green for correct, red for wrong

**Scheduler integration** in `dashboard/app.py`:
```python
scheduler.add_job(
    scheduled_grade_predictions,
    'cron',
    hour=1,   # 1 AM — all games finished
    minute=0,
    id='grade_predictions',
)
```

**Dashboard download button** (add to Today's Games page header):
```python
html.A("⬇ Download Report", href="/download-report",
       style={...button style...})

# Flask route on app.server:
@app.server.route("/download-report")
def download_report():
    from utils.prediction_tracker import export_to_excel
    path = export_to_excel()
    return send_file(path, as_attachment=True)
```

---

## AREA 3 — Best Props Enhancements

### 3A. Surface 100% Hit Rate Props

**File:** `utils/props_cache.py` — `_compute_main_page_props()`

**Change:** After building `props_data`, add a `"is_lock"` field:
```python
prop["is_lock"] = (prop["hit_rate"] == 1.0 and prop["total"] >= 5)
```

**UI:** In `dashboard/app.py` prop card renderer, add a "🔒 LOCK" badge when `is_lock=True`.
Sort locks to top of list regardless of EV (within their stat category).

---

### 3B. Weighted Combo Props with Smart Filtering

**File:** `utils/props_cache.py` — `_compute_main_page_props()`

**Current:** Only single stats (PTS, AST, REB, FG3M) are processed.

**New:** Add combo prop evaluation with contribution weighting.

**Weight constants:**
```python
STAT_WEIGHTS = {"PTS": 1.0, "REB": 0.6, "AST": 0.7, "FG3M": 0.8}
MIN_MEANINGFUL_AVG = {"PTS": 12.0, "REB": 4.0, "AST": 3.0}
```

**Smart combo filtering logic:**
```python
def _is_meaningful_combo(stat_avgs: dict) -> bool:
    """
    Return True only if all stats in combo are meaningful.
    If PTS >> (REB + AST), recommend PTS-only instead.
    """
    pts = stat_avgs.get("PTS", 0)
    reb = stat_avgs.get("REB", 0)
    ast = stat_avgs.get("AST", 0)

    # PRA only meaningful if player averages in all 3 categories
    if "REB" in stat_avgs and reb < MIN_MEANINGFUL_AVG["REB"]:
        return False
    if "AST" in stat_avgs and ast < MIN_MEANINGFUL_AVG["AST"]:
        return False

    # If player is a pure scorer (PTS > 5x combined others), skip combo
    if pts > 0 and (reb + ast) > 0 and pts / (reb + ast) > 4.0:
        return False

    return True
```

**Combo prop line setting:**
```python
# Weighted line — gives more weight to higher-value stats
weighted_avg = sum(avg * STAT_WEIGHTS[s] for s, avg in stat_avgs.items())
             / sum(STAT_WEIGHTS[s] for s in stat_avgs)
combo_line = round(weighted_avg * 2) / 2
```

**Combo display label:**
```python
# e.g. "Pts+Reb — 28.5 (avg 29.2 L10)"
```

---

## AREA 4 — Player Analysis Tab Fixes

### 4A. Stat Filter Tab Padding Fix

**File:** `dashboard/assets/custom.css`

**Problem:** `.tab` padding is uneven; hover highlight shifts left.

**Fix:**
```css
.tab-group .tab {
    padding: 6px 16px;           /* equal horizontal padding */
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;     /* center text in highlight */
    min-width: 52px;
    box-sizing: border-box;
}

.tab-group .tab:hover {
    background: rgba(255, 255, 255, 0.07);
    /* Remove any transform: translateX that may exist */
    transform: none;
}
```

Search for any `transform: translateX` or `margin-left` on `.tab:hover` and remove it.

---

### 4B. Player Analysis Injury Status Fix

**File:** `dashboard/app.py` — sidebar injury tab callback (search for `sidebar-tab` output with `injuries` case)

**Problem:** Shows every player as "Active" regardless of actual status.

**Find the callback** that renders injury content when `sidebar-tab == "injuries"`.

**Current (broken):**
```python
# Likely just returns static "Active" for every player
```

**Fix:**
```python
from utils.injury_news import get_player_injury_status

def render_injury_sidebar(player_name):
    status_data = get_player_injury_status(player_name)
    status = status_data.get("status", "ACTIVE")
    reason = status_data.get("reason", "")

    status_colors = {
        "OUT":          "#f43f5e",
        "DOUBTFUL":     "#f97316",
        "QUESTIONABLE": "#f59e0b",
        "PROBABLE":     "#a3e635",
        "ACTIVE":       "#22c55e",
    }
    color = status_colors.get(status, "#22c55e")

    return html.Div([
        html.Div(status, style={"color": color, "fontWeight": "700", "fontSize": "1rem"}),
        html.Div(reason or "No injury report", style={"color": "var(--text-muted)", "fontSize": "0.85rem", "marginTop": "4px"}),
        html.Div(f"Updated: {status_data.get('checked_at', 'N/A')}",
                 style={"color": "var(--text-muted)", "fontSize": "0.75rem", "marginTop": "8px"}),
    ])
```

---

### 4C. Supporting Stats Chart Fixes

**File:** `dashboard/app.py` — `update_shooting_breakdown_chart()` callback (around Line 2717)

**Problems:**
1. MIN tab click doesn't show minutes — likely `MIN` not in the stat mapping
2. All bars same gray color
3. X-axis crowded with too many games

**Fixes:**

**1. Limit to last 20 games max:**
```python
# In chart callback, slice player_df:
chart_df = player_df.sort_values("_date", ascending=False).head(20)
# Use head(10) if more than 15 games would crowd x-axis
n_games = min(20, len(player_df))
chart_df = player_df.sort_values("_date", ascending=False).head(n_games)
```

**2. Distinct colors per stat:**
```python
STAT_COLORS = {
    "PTS":  "#14b8a6",   # teal
    "REB":  "#a78bfa",   # violet
    "AST":  "#f97066",   # coral
    "FG3M": "#60a5fa",   # blue
    "FGM":  "#fbbf24",   # amber
    "FG":   "#fbbf24",   # amber (shooting %)
    "FG3":  "#60a5fa",   # blue (3pt %)
    "FT":   "#34d399",   # green
    "MIN":  "#94a3b8",   # slate (minutes)
    "PF":   "#fb923c",   # orange
}
bar_color = STAT_COLORS.get(selected_stat, "#64748b")
```

**3. Fix MIN tab:**
Check that `"MIN"` is in the `stat_cards` list in `update_supporting_stats_cards()`.
If stat value is in `supporting_stats` dict as `"MIN"`, ensure the chart callback maps `"MIN"` → `player_df["MIN"]` column correctly.
```python
# In chart data builder:
if selected_stat == "MIN":
    y_vals = pd.to_numeric(chart_df["MIN"], errors="coerce")
    y_label = "Minutes"
elif selected_stat in ("FG", "FG3", "FT"):
    # These are percentages
    col_map = {"FG": "FG_PCT", "FG3": "FG3_PCT", "FT": "FT_PCT"}
    y_vals = chart_df[col_map[selected_stat]] * 100
    y_label = f"{selected_stat} %"
else:
    y_vals = chart_df[selected_stat]
    y_label = selected_stat
```

**4. X-axis label fix:**
```python
fig.update_layout(
    xaxis=dict(
        tickangle=-45 if n_games > 10 else 0,
        tickfont=dict(size=10),
        nticks=min(n_games, 15),  # never more than 15 ticks
    ),
    margin=dict(b=60 if n_games > 10 else 40),  # bottom margin for rotated labels
)
```

---

## Implementation Order & File Changes

| Priority | Area | File(s) | Effort |
|----------|------|---------|--------|
| 1 | Supporting stats chart fixes (4C) | app.py | Low |
| 2 | Tab padding fix (4A) | custom.css | Low |
| 3 | Injury status in player analysis (4B) | app.py | Low |
| 4 | Injury-aware predictor (1A) | game_predictor.py | Medium |
| 5 | Pick reasoning/explanations (1B) | game_predictor.py, app.py | Medium |
| 6 | Game time fix (1C) | app.py | Low |
| 7 | Injury panel → dropdown (1D) | app.py | Low |
| 8 | Prediction tracker + Excel (2A) | NEW prediction_tracker.py, app.py | High |
| 9 | 100% lock props (3A) | props_cache.py, app.py | Low |
| 10 | Weighted combo props (3B) | props_cache.py | Medium |

## Key Files Summary

| File | Operations |
|------|-----------|
| `utils/game_predictor.py` | Add `_apply_injury_penalty()`, modify `predict_game()` signature, add `build_pick_reasoning()` |
| `utils/prediction_tracker.py` | **NEW** — save/grade/export predictions |
| `utils/props_cache.py` | Add lock detection, add combo prop evaluation with weights |
| `dashboard/app.py` | Fix game time display, fix injury sidebar, fix chart callback, add download route, add reasoning UI |
| `dashboard/assets/custom.css` | Fix tab padding and hover alignment |

## SESSION_IDs
- CODEX_SESSION: N/A (direct Claude planning)
- GEMINI_SESSION: N/A (direct Claude planning)
