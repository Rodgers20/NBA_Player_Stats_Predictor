# Implementation Plan: Bet History — Last 10 Games Over/Under

## Task Type
- [x] Frontend + Backend (single file: `dashboard/app.py`)

## Requirement
Replace the current `create_prop_bet_history_card()` which pulls from props_cache with a
real-data card showing the **last 10 games** for the selected player and stat, with an
Over/Under pill per game based on the player's **season average** as the line.

Example: Player averages 20 PPG
- Game 1: vs LAL → 22 pts → ✓ OVER  (teal pill)
- Game 2: vs GSW → 17 pts → ✕ UNDER (rose pill)

When user switches to AST tab, show last 10 games AST values vs AST season avg.
Same for REB, FG3M, etc.

---

## Technical Solution

### Data source
`DF` (the global games DataFrame) — already in scope inside `create_prop_bet_history_card()`.
Columns used: `PLAYER_NAME`, `_date`, `PTS`, `REB`, `AST`, `FG3M`, `MATCHUP` (or `TEAM_ABBREVIATION`)

### Stat mapping
| Tab `stat` value | DF column | Display label |
|------------------|-----------|---------------|
| `PTS`   | `PTS`  | Points |
| `AST`   | `AST`  | Assists |
| `REB`   | `REB`  | Rebounds |
| `FG3M`  | `FG3M` | 3-Pointers |

### Line (threshold)
Season average of the selected stat across all games in DF for that player.

### Over/Under logic
```
actual = row[stat_col]
is_over = actual >= season_avg   # >= means "hit the line or better"
```

### Opponent display
`MATCHUP` column contains strings like `"LAL vs. GSW"` or `"LAL @ GSW"`.
Extract the opponent abbreviation: strip the player's team side.

---

## Implementation Steps

### Step 1 — Rewrite `create_prop_bet_history_card(player_name, stat)`

Replace entire function body (keep function signature — it's already called correctly).

```python
def create_prop_bet_history_card(player_name, stat):
    """Last 10 games over/under season average for selected stat."""

    # ── Stat config ──────────────────────────────────────────────────
    STAT_COL_MAP = {
        "PTS": ("PTS", "Points"),
        "AST": ("AST", "Assists"),
        "REB": ("REB", "Rebounds"),
        "FG3M": ("FG3M", "3-Pointers"),
        "PTS+REB": ("PTS", "Points"),   # fallback to PTS for combos
        "PTS+AST": ("PTS", "Points"),
        "PTS+REB+AST": ("PTS", "Points"),
    }
    stat_col, stat_label = STAT_COL_MAP.get(stat, ("PTS", "Points"))

    # ── Pull data ────────────────────────────────────────────────────
    player_df = DF[DF["PLAYER_NAME"] == player_name].copy()
    if player_df.empty or stat_col not in player_df.columns:
        return _empty_bet_history_card(stat_label)

    player_df = player_df.sort_values("_date", ascending=False)

    season_avg = player_df[stat_col].mean()   # full season avg = the "line"
    last10 = player_df.head(10)

    # ── Build rows ───────────────────────────────────────────────────
    rows = []
    for _, row in last10.iterrows():
        actual = row.get(stat_col, 0)
        if actual is None or (hasattr(actual, '__class__') and actual.__class__.__name__ == 'float' and actual != actual):
            continue  # skip NaN
        actual = float(actual)
        is_over = actual >= season_avg

        # Opponent: MATCHUP looks like "BOS vs. MIA" or "BOS @ MIA"
        matchup = str(row.get("MATCHUP", ""))
        team = str(row.get("TEAM_ABBREVIATION", ""))
        if " vs. " in matchup:
            parts = matchup.split(" vs. ")
            opp = parts[1] if parts[0].strip() == team else parts[0]
        elif " @ " in matchup:
            parts = matchup.split(" @ ")
            opp = parts[1] if parts[0].strip() == team else parts[0]
        else:
            opp = matchup[-3:] if len(matchup) >= 3 else "???"
        opp = opp.strip()

        # Date label
        try:
            date_str = row["_date"].strftime("%-m/%-d") if hasattr(row["_date"], "strftime") else str(row["_date"])[:5]
        except Exception:
            date_str = ""

        pill_style = {
            "display": "inline-flex", "alignItems": "center", "gap": "4px",
            "borderRadius": "6px", "padding": "3px 8px",
            "fontSize": "0.7rem", "fontWeight": "600", "flexShrink": "0",
        }
        if is_over:
            pill_style.update({
                "background": "rgba(45,212,191,0.12)", "border": "1px solid rgba(45,212,191,0.25)",
                "color": "#2DD4BF",
            })
            pill_content = [html.Span("✓"), html.Span("OVER")]
        else:
            pill_style.update({
                "background": "rgba(251,113,133,0.12)", "border": "1px solid rgba(251,113,133,0.25)",
                "color": "#FB7185",
            })
            pill_content = [html.Span("✕"), html.Span("UNDER")]

        rows.append(html.Div([
            # Left: date + opponent
            html.Div([
                html.Span(date_str, style={"fontSize": "0.65rem", "color": "#64748b", "marginRight": "4px"}),
                html.Span(f"vs {opp}", style={"fontSize": "0.78rem", "color": "#94a3b8", "fontWeight": "500"}),
            ], style={"display": "flex", "alignItems": "center", "minWidth": "0"}),

            # Right: actual value + pill
            html.Div([
                html.Span(f"{actual:.0f}", style={
                    "fontSize": "0.92rem", "fontWeight": "700",
                    "color": "#2DD4BF" if is_over else "#FB7185",
                    "marginRight": "6px",
                }),
                html.Div(pill_content, style=pill_style),
            ], style={"display": "flex", "alignItems": "center", "flexShrink": "0"}),
        ], style={
            "display": "flex", "alignItems": "center", "justifyContent": "space-between",
            "padding": "8px 4px", "borderBottom": "1px solid rgba(30,41,59,0.4)",
        }))

    if not rows:
        return _empty_bet_history_card(stat_label)

    # ── Card header ──────────────────────────────────────────────────
    header = html.Div([
        html.Span("Bet History", className="matchup-section-title",
                  style={"marginBottom": "0", "color": "#f1f5f9", "fontWeight": "700"}),
        html.Div([
            html.Span("Avg: ", style={"color": "#64748b", "fontSize": "0.7rem"}),
            html.Span(f"{season_avg:.1f} {stat_label[:3].upper()}", style={
                "color": "#94a3b8", "fontSize": "0.75rem", "fontWeight": "600",
            }),
        ]),
    ], style={"display": "flex", "justifyContent": "space-between",
              "alignItems": "center", "marginBottom": "10px"})

    return html.Div([header, html.Div(rows)], className="analysis-card")


def _empty_bet_history_card(stat_label="Points"):
    return html.Div([
        html.Span("Bet History", className="matchup-section-title",
                  style={"color": "#f1f5f9", "fontWeight": "700"}),
        html.Div("No game data available.",
                 style={"fontSize": "0.8rem", "color": "#6b7280", "paddingTop": "8px"}),
    ], className="analysis-card")
```

### Step 2 — Remove unused imports in the old function
The old function imported `get_cached_props` and used `defaultdict`. Remove those from inside the function body (they're local imports — no module-level side effects).

### Step 3 — CSS: row hover
Add to `custom.css`:
```css
/* Bet history rows */
.bet-history-row:hover {
  background: rgba(30,41,59,0.3);
  border-radius: 6px;
}
```
(Optional — low priority)

---

## Key Files

| File | Operation | Lines |
|------|-----------|-------|
| `dashboard/app.py` | Modify | `create_prop_bet_history_card()` ~L4187–4285 |
| `dashboard/assets/custom.css` | Optional | Hover style for rows |

---

## Edge Cases
- Player with fewer than 10 games → show however many exist (head(10) handles naturally)
- Stat not in DF columns → fallback to PTS or show empty card
- NaN values → skip that game row
- Combo stats (PTS+REB+AST) → fallback to PTS column

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `_date` not a datetime | Wrap in try/except, fall back to raw string |
| MATCHUP format varies | Handle both `vs.` and `@` formats, strip to last 3 chars as final fallback |
| `stat_col` not in DF | Guard check before `.mean()` |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
