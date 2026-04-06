# Implementation Plan: Games Page Redesign

## Feature Summary
Redesign the Today's Games page to match the reference screenshot:
- "NBA Game Predictions" title at top left
- Date display: "< MAR 23, 2026 (TODAY) >" centered
- Horizontal scrollable game selector tabs (team logos + time)
- Single large selected-game panel with 3-column layout:
  - Left: AWAY TEAM STATS (Off Rating, Def Rating, Recent ATS, Injury Report)
  - Center: Large team logos + donut win-probability chart
  - Right: HOME TEAM STATS (same)
- Bottom of panel: "AI PREDICTION & RECOMMENDED PLAY" with pick + confidence bar

---

## Architecture

### Data Flow
```
create_todays_games_page()
  → computes ALL game data (odds, predictor, form, injuries) for all games
  → stores serialized list in dcc.Store("games-page-data")
  → renders game-selector bar + empty panel container

update_game_panel(selected_idx, games_data)
  → callback triggered by clicking any game tab
  → deserializes the selected game's data
  → builds and returns the big 3-col panel HTML
```

### Why store + callback (not pre-render all)
Pre-rendering 10 game panels at once wastes DOM memory. The store approach renders only the selected game — fast tab switching with no re-computation since data is already in the store.

---

## Implementation Steps

### Step 1 — CSS additions
**File**: `dashboard/assets/custom.css`

```css
/* Game selector bar */
.game-selector-bar {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 20px;
    scrollbar-width: none;
}
.game-selector-bar::-webkit-scrollbar { display: none; }

/* Individual game selector pill */
.game-tab-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 9999px;
    border: 1px solid rgba(255,255,255,0.1);
    background: #1c2131;
    color: #9ca3af;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    flex-shrink: 0;
    transition: all 0.15s ease;
}
.game-tab-pill:hover {
    border-color: rgba(255,255,255,0.2);
    color: #f0f4ff;
}
.game-tab-pill.active {
    background: #ffffff;
    color: #0f1623;
    border-color: transparent;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.game-tab-pill img { width: 22px; height: 22px; object-fit: contain; }

/* Main game panel — 3-column */
.game-panel {
    background: #1c2131;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 32px 28px 24px;
    max-width: 1100px;
    margin: 0 auto;
}
.game-panel-3col {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 24px;
    align-items: start;
    margin-bottom: 24px;
}

/* Team stats column */
.team-stats-col { display: flex; flex-direction: column; gap: 14px; }
.team-stats-title {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: #9ca3af;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* Stat row with progress bar */
.stat-row { display: flex; flex-direction: column; gap: 4px; }
.stat-row-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.stat-row-label { font-size: 0.82rem; color: #9ca3af; display: flex; align-items: center; gap: 8px; }
.stat-row-value { font-size: 0.88rem; font-weight: 700; color: #f0f4ff; }
.stat-progress-track {
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
    overflow: hidden;
}
.stat-progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}
.stat-progress-blue  { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.stat-progress-green { background: linear-gradient(90deg, #22c55e, #4ade80); }
.stat-progress-amber { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.stat-progress-rose  { background: linear-gradient(90deg, #f43f5e, #fb7185); }

/* Center column — logos + donut */
.game-center-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 300px;
}
.team-logos-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 32px;
    margin-bottom: 16px;
}
.team-logo-large { width: 90px; height: 90px; object-fit: contain; }
.vs-text {
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #6b7280;
}

/* AI Prediction footer */
.ai-prediction-section {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 28px;
    text-align: center;
}
.ai-prediction-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #9ca3af;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.ai-prediction-pick {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f0f4ff;
    letter-spacing: -0.02em;
    margin-bottom: 14px;
}
.confidence-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 4px;
    overflow: hidden;
    max-width: 500px;
    margin: 0 auto 8px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #f97316, #22c55e);
}
.confidence-level-text { font-size: 0.88rem; color: #9ca3af; }
.confidence-level-text .conf-value { font-weight: 700; }
.conf-high  { color: #22c55e; }
.conf-med   { color: #f59e0b; }
.conf-low   { color: #94a3b8; }

/* Injury report inside stats col */
.injury-report-section { margin-top: 4px; }
.injury-report-title {
    font-size: 0.75rem; font-weight: 700; color: #9ca3af;
    display: flex; align-items: center; gap: 6px; margin-bottom: 6px;
}
.injury-entry { font-size: 0.78rem; color: #d1d5db; margin-bottom: 3px; }
.injury-status-q  { color: #f59e0b; font-weight: 600; }
.injury-status-out { color: #f43f5e; font-weight: 600; }
.injury-status-p  { color: #22c55e; font-weight: 600; }

@media (max-width: 860px) {
    .game-panel-3col { grid-template-columns: 1fr; }
    .game-center-col { min-width: 0; }
}
```

---

### Step 2 — Restructure `create_todays_games_page()`
**File**: `dashboard/app.py` ~L1171

Serialize each game's computed data into a JSON-serializable dict stored in `dcc.Store`. No more building HTML cards in the loop.

```python
def create_todays_games_page():
    games     = get_todays_games()
    game_odds = get_game_odds()
    GAME_PREDICTOR.refresh(get_global_df())
    _ABBR_FIX = {"SAS": "SAN"}

    if games.empty:
        return html.Div([...no games UI...])

    # ── Serialise all game data into a list ───────────────────────────────
    games_data = []
    _predictions_for_tracker = []

    for _, game in games.iterrows():
        home_team = ...
        away_team = ...
        # (all existing computation: odds, predictor, form, injuries)
        # Build a JSON-serializable dict instead of HTML:
        entry = {
            "home":        home_team,
            "away":        away_team,
            "home_logo":   get_team_logo_url(home_team),
            "away_logo":   get_team_logo_url(away_team),
            "game_time":   _format_game_time(game.get("GAME_TIME", "")),
            "home_off":    round(home_form.get("rolling_ppg") or home_season.get("ppg", 0), 1),
            "away_off":    round(away_form.get("rolling_ppg") or away_season.get("ppg", 0), 1),
            "home_def":    round(home_form.get("rolling_opp_ppg") or home_season.get("opp_ppg", 0), 1),
            "away_def":    round(away_form.get("rolling_opp_ppg") or away_season.get("opp_ppg", 0), 1),
            "home_wins":   home_form.get("wins", 0),
            "home_losses": home_form.get("losses", 0),
            "away_wins":   away_form.get("wins", 0),
            "away_losses": away_form.get("losses", 0),
            "home_win_pct": home_win_pct,   # 0-100 float
            "away_win_pct": away_win_pct,
            "spread_line":  spread_home_line,
            "total_line":   total_line,
            "spread_pick":  pick.get("spread_pick"),
            "spread_team":  pick.get("spread_team"),
            "spread_conf":  pick.get("spread_confidence"),
            "total_pick":   pick.get("total_pick"),
            "total_conf":   pick.get("total_confidence"),
            "winner_pick":  pick.get("winner_pick"),
            "winner_conf":  pick.get("winner_confidence"),
            "pred_home":    pred_home,
            "pred_away":    pred_away,
            "pred_total":   pred_total,
            "home_injuries": [...],  # [{name, status}]
            "away_injuries": [...],
        }
        games_data.append(entry)

    # Save predictions tracker (same as before)
    ...

    # ── Build game selector pills ─────────────────────────────────────────
    game_pills = []
    for i, g in enumerate(games_data):
        game_pills.append(html.Div([
            html.Img(src=g["away_logo"], style={"width":"20px","height":"20px","objectFit":"contain"}) if g["away_logo"] else None,
            html.Span(f"{g['away']} vs {g['home']}"),
            html.Span(f" - {g['game_time']}", style={"opacity":"0.7","fontWeight":"400"}),
        ],
            id={"type": "game-tab-pill", "index": i},
            n_clicks=0,
            className="game-tab-pill" + (" active" if i == 0 else ""),
        ))

    return html.Div([
        # Page header
        html.Div([
            html.Div("NBA Game Predictions",
                     style={"fontSize":"1.8rem","fontWeight":"800","color":"#f0f4ff","letterSpacing":"-0.02em"}),
            html.Div(datetime.now().strftime("%B %d, %Y"),
                     style={"fontSize":"0.9rem","color":"#6b7280","marginTop":"2px"}),
        ], style={"marginBottom":"20px"}),

        # Game selector bar
        html.Div(game_pills, className="game-selector-bar"),

        # Selected game index store
        dcc.Store(id="selected-game-idx", data=0),
        dcc.Store(id="games-page-data", data=games_data),

        # Panel container (populated by callback)
        html.Div(id="game-panel-container"),

    ], style={"maxWidth":"1200px","margin":"0 auto","padding":"2rem"})
```

---

### Step 3 — Game tab click callback
**File**: `dashboard/app.py`

```python
@callback(
    Output("selected-game-idx", "data"),
    Input({"type": "game-tab-pill", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_game_tab(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        raise PreventUpdate
    return triggered["index"]
```

---

### Step 4 — Panel render callback + donut chart
**File**: `dashboard/app.py`

```python
@callback(
    Output("game-panel-container", "children"),
    [Input("selected-game-idx", "data"),
     Input("games-page-data", "data")],
)
def update_game_panel(selected_idx, games_data):
    if not games_data:
        return html.Div("No games data", style={"textAlign":"center","padding":"40px","color":"#6b7280"})
    idx = selected_idx or 0
    g = games_data[idx]

    # ── Donut win-probability chart ───────────────────────────────────────
    home_wp = g["home_win_pct"]
    away_wp = 100 - home_wp
    fig = go.Figure(data=[go.Pie(
        values=[home_wp, away_wp],
        labels=[g["home"], g["away"]],
        hole=0.68,
        marker_colors=["#14b8a6", "#1e3a5f"],
        direction="clockwise",
        textinfo="none",
        hoverinfo="label+percent",
    )])
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        annotations=[dict(
            text=f"<b>{g['home']}</b><br>{home_wp:.0f}%",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#f0f4ff"),
        )]
    )

    # ── Stat progress bar helper ──────────────────────────────────────────
    def stat_row(icon, label, value, max_val, color_class):
        pct = min(100, max(0, (value / max_val) * 100)) if max_val > 0 else 0
        return html.Div([
            html.Div([
                html.Span(f"{icon}  ", style={"marginRight":"4px"}),
                html.Span(label, className="stat-row-label"),
                html.Span(str(value), className="stat-row-value"),
            ], className="stat-row-header"),
            html.Div(html.Div(className=f"stat-progress-fill {color_class}",
                     style={"width": f"{pct:.0f}%"}),
                     className="stat-progress-track"),
        ], className="stat-row")

    # ── Injury rows ───────────────────────────────────────────────────────
    def injury_rows(injuries):
        if not injuries:
            return html.Div("No significant injuries", style={"fontSize":"0.78rem","color":"#6b7280"})
        rows = []
        for inj in injuries[:4]:
            status = inj.get("status", "")
            cls = "injury-status-q" if "questionable" in status.lower() else \
                  "injury-status-out" if status.lower() in ("out","doubtful") else "injury-status-p"
            rows.append(html.Div([
                html.Span(inj.get("name",""), style={"color":"#d1d5db"}),
                html.Span(f" ({status})", className=cls),
            ], className="injury-entry"))
        return html.Div(rows)

    # ── AI Prediction text ────────────────────────────────────────────────
    pick_parts = []
    spread_pick = g.get("spread_pick")
    spread_team = g.get("spread_team")
    spread_line = g.get("spread_line")
    total_pick  = g.get("total_pick")
    total_line  = g.get("total_line")

    if spread_pick and spread_team and spread_line is not None:
        sign = "+" if spread_line > 0 else ""
        pick_parts.append(f"{spread_team} {sign}{spread_line:.1f}")
    elif g.get("pred_home") and g.get("pred_away"):
        model_spread = g["pred_home"] - g["pred_away"]
        leader = g["home"] if model_spread > 0 else g["away"]
        sign = "+" if model_spread > 0 else ""
        pick_parts.append(f"{leader} {sign}{model_spread:.1f}")

    if total_pick and total_line:
        pick_parts.append(f"{total_pick} {total_line:.1f}")
    elif g.get("pred_total"):
        pick_parts.append(f"TOTAL {g['pred_total']:.1f}")

    pick_text = " | ".join(pick_parts) if pick_parts else "Model processing..."

    # Confidence
    conf_raw = g.get("winner_conf", "LOW")
    conf_pct = {"HIGH": 85, "MEDIUM": 65, "LOW": 45}.get(conf_raw, 50)
    conf_cls = "conf-high" if conf_raw == "HIGH" else "conf-med" if conf_raw == "MEDIUM" else "conf-low"

    # ── Assemble panel ────────────────────────────────────────────────────
    home_off = g.get("home_off", 0) or 0
    away_off = g.get("away_off", 0) or 0
    home_def = g.get("home_def", 0) or 0
    away_def = g.get("away_def", 0) or 0
    max_off = max(home_off, away_off, 120) or 120
    max_def = max(home_def, away_def, 120) or 120

    return html.Div([
        html.Div([
            # ── LEFT: Away team stats ─────────────────────────────────────
            html.Div([
                html.Div(f"{g['away']} STATS", className="team-stats-title"),
                stat_row("⚡", "Offensive Rating:", home_off, max_off, "stat-progress-blue"),
                stat_row("🛡", "Defensive Rating:", home_def, max_def, "stat-progress-rose"),
                stat_row("📊", f"Recent ATS (L10):", f"{g['away_wins']}-{g['away_losses']}", 1, "stat-progress-amber"),
                html.Div([
                    html.Div(["🏥  ", html.Span("Injury Report:", style={"fontWeight":"700"})],
                             className="injury-report-title"),
                    injury_rows(g.get("away_injuries", [])),
                ], className="injury-report-section"),
            ], className="team-stats-col"),

            # ── CENTER: Logos + donut ─────────────────────────────────────
            html.Div([
                html.Div([
                    html.Img(src=g["away_logo"], className="team-logo-large") if g["away_logo"] else html.Div(g["away"]),
                    html.Div("VS", className="vs-text"),
                    html.Img(src=g["home_logo"], className="team-logo-large") if g["home_logo"] else html.Div(g["home"]),
                ], className="team-logos-row"),
                dcc.Graph(figure=fig, config={"displayModeBar": False},
                          style={"width":"240px","height":"220px"}),
                html.Div([
                    html.Span(f"{g['away']} {away_wp:.0f}%  Win Prob",
                              style={"fontSize":"0.78rem","color":"#6b7280","marginRight":"16px"}),
                    html.Span(f"{g['home']} {home_wp:.0f}%  Win Prob",
                              style={"fontSize":"0.78rem","color":"#14b8a6"}),
                ], style={"textAlign":"center","marginTop":"8px"}),
            ], className="game-center-col"),

            # ── RIGHT: Home team stats ────────────────────────────────────
            html.Div([
                html.Div(f"{g['home']} STATS", className="team-stats-title",
                         style={"textAlign":"right"}),
                stat_row("⚡", "Offensive Rating:", away_off, max_off, "stat-progress-blue"),
                stat_row("🛡", "Defensive Rating:", away_def, max_def, "stat-progress-rose"),
                stat_row("📊", f"Recent ATS (L10):", f"{g['home_wins']}-{g['home_losses']}", 1, "stat-progress-green"),
                html.Div([
                    html.Div(["🏥  ", html.Span("Injury Report:", style={"fontWeight":"700"})],
                             className="injury-report-title"),
                    injury_rows(g.get("home_injuries", [])),
                ], className="injury-report-section"),
            ], className="team-stats-col"),

        ], className="game-panel-3col"),

        # ── AI Prediction footer ──────────────────────────────────────────
        html.Div([
            html.Div("AI PREDICTION & RECOMMENDED PLAY", className="ai-prediction-label"),
            html.Div(pick_text, className="ai-prediction-pick"),
            html.Div(html.Div(className="confidence-bar-fill",
                     style={"width": f"{conf_pct}%"}),
                     className="confidence-bar-track"),
            html.Div([
                "Confidence Level: ",
                html.Span(conf_raw, className=f"conf-value {conf_cls}"),
                html.Span(f" ({conf_pct}%)", style={"color":"#9ca3af"}),
            ], className="confidence-level-text"),
        ], className="ai-prediction-section"),

    ], className="game-panel")
```

---

### Step 5 — Tab active state callback
**File**: `dashboard/app.py`

Update pill classNames when selection changes:

```python
@callback(
    Output({"type": "game-tab-pill", "index": ALL}, "className"),
    Input("selected-game-idx", "data"),
    State({"type": "game-tab-pill", "index": ALL}, "id"),
)
def update_game_tab_classes(selected_idx, pill_ids):
    return [
        "game-tab-pill active" if p["index"] == selected_idx else "game-tab-pill"
        for p in (pill_ids or [])
    ]
```

---

### Step 6 — Injury data serialisation
In the loop inside `create_todays_games_page()`, build injury lists:

```python
def _serialise_injuries(team_abbr):
    try:
        inj_list = get_team_injuries(team_abbr)
        return [{"name": i.get("name",""), "status": i.get("status","")}
                for i in inj_list if i.get("status","").lower() not in ("","active","healthy")][:5]
    except Exception:
        return []

entry["home_injuries"] = _serialise_injuries(home_team_internal)
entry["away_injuries"] = _serialise_injuries(away_team_internal)
```

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/assets/custom.css` | Add | Game selector, panel, stat-row, confidence-bar CSS classes |
| `dashboard/app.py:1171` | Rewrite | `create_todays_games_page()` — serialise data, render selector + stores |
| `dashboard/app.py` | Add | `select_game_tab()` callback — pattern-matching pill click → store |
| `dashboard/app.py` | Add | `update_game_panel()` callback — render big panel from stored data |
| `dashboard/app.py` | Add | `update_game_tab_classes()` callback — keep active pill highlighted |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `games_data` store too large for Dash (>5MB limit) | Keep only what's needed; remove large dicts like wl_list (keep only wins/losses counts) |
| `get_team_injuries()` slow for 20 teams (10 games × 2) | Call once per team, cache results with a dict in the loop |
| Win probability calculation | Use `prediction.get("winner_confidence")` + `prediction.get("winner_margin")` to compute pct: HIGH=80-90%, MEDIUM=60-75%, LOW=50-55% |
| Prediction tracker fire-and-forget thread — must still fire | Keep existing `_predictions_for_tracker` append logic in loop, fire thread after loop |
| `game-tab-pill` active callback needs `State` for IDs | Use `State({"type":"game-tab-pill","index":ALL},"id")` |
| Progress bar width % must be valid CSS | Clamp to 0-100 before rendering |

---

## SESSION_ID
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
