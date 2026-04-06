# Implementation Plan: Best Props Player Performance Panel

## Feature Summary
When a user clicks a player's photo or name on the Best Props page, a panel slides
in on the **left side** of the screen showing a bar chart of that player's game-by-game
performance. The default stat shown is the one the model picked the player for (e.g.
FG3M for Matisse Thybulle). The user can then switch between PTS, AST, REB, FG3M,
3PM, STL, BLK stat tabs — same as the Player Analysis page.

---

## Architecture

### Layout change (currently: single column → flex row)
```
[ props-player-panel (left, 420px, hidden by default) ] [ props-list (right, flex: 1) ]
```
- Panel slides in when a player is selected; props list shifts right.
- Panel has a close "×" button to dismiss.

### Data flow
1. User clicks photo or name on a prop card.
2. `prop-player-name` / `prop-player-photo` click callback → writes
   `{"player": "Matisse Thybulle", "default_stat": "FG3M"}` to `dcc.Store(id="props-panel-store")`.
3. Panel render callback reads the store → draws chart.
4. Stat tab click callback → re-renders chart with new stat (no page navigation).

---

## Implementation Steps

### Step 1 — Layout: add panel placeholder to `create_best_props_page()`
**File**: `dashboard/app.py` → `create_best_props_page()` ~L1664

Currently returns:
```python
html.Div([
    html.Div([...filters, props-list...], style={"maxWidth":"800px","margin":"0 auto"})
])
```

Change to flex row:
```python
html.Div([
    # LEFT: player chart panel (hidden initially)
    html.Div(id="props-player-panel", style={
        "width": "0px",
        "minWidth": "0px",
        "overflow": "hidden",
        "transition": "width 0.3s ease, min-width 0.3s ease",
        "flexShrink": "0",
    }),
    # RIGHT: existing props content
    html.Div([
        ...existing content...,
        dcc.Store(id="props-panel-store", data=None),   # NEW
        dcc.Store(id="props-panel-stat", data=None),    # NEW (active stat tab)
    ], style={"flex":"1","minWidth":"0"}),
], style={"display":"flex","gap":"16px","alignItems":"flex-start"})
```

---

### Step 2 — Make player photo clickable in prop card builder
**File**: `dashboard/app.py` → `update_best_props_main()` callback, prop card builder ~L2195

Currently photo is a plain `html.Img`. Change to a clickable `html.Div`:
```python
# Encode player|stat in the index so callback knows which stat to default to
html.Div(
    html.Img(src=player_photo, style={...existing styles...}) if player_photo else html.Div(...),
    id={"type": "prop-player-photo", "index": f"{prop['player']}|{prop['stat']}"},
    n_clicks=0,
    style={"marginRight":"14px","flexShrink":"0","cursor":"pointer"},
)
```

Also update the player name span to encode the stat:
```python
html.Span(
    prop.get("player",""),
    id={"type": "prop-player-name", "index": f"{prop['player']}|{prop['stat']}"},
    n_clicks=0,
    style={...existing styles...}
)
```

> Note: Currently clicking `prop-player-name` navigates to Player Analysis.
> Change that callback (`select_player_from_prop`) so it no longer routes — instead,
> the new panel callback handles the click.

---

### Step 3 — Remove navigation from existing `select_player_from_prop` callback
**File**: `dashboard/app.py` ~L1989

Currently outputs `player-dropdown` value + URL navigation. This must be decoupled:
- Keep the URL navigation on a **separate** "Go to full analysis" button inside the panel.
- The prop-player-name click should only open the panel, not navigate.

Change the callback to update `props-panel-store` instead of navigating:
```python
@callback(
    Output("props-panel-store", "data"),
    Input({"type": "prop-player-name", "index": ALL}, "n_clicks"),
    Input({"type": "prop-player-photo", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def open_props_player_panel(name_clicks, photo_clicks):
    triggered = ctx.triggered_id
    if not triggered or not any(name_clicks + photo_clicks):
        raise PreventUpdate
    raw_index = triggered["index"]          # e.g. "Matisse Thybulle|FG3M"
    parts = raw_index.split("|", 1)
    player = parts[0]
    default_stat = parts[1] if len(parts) > 1 else "PTS"
    # Map FG3M → "3PM" display label if needed
    return {"player": player, "default_stat": default_stat, "active_stat": default_stat}
```

---

### Step 4 — New helper: `_build_props_panel_chart(player_name, stat, n_games=10)`
**File**: `dashboard/app.py` (add near the other chart helpers ~L2938)

Reuses the same chart-building logic as `update_main_chart`:
```python
def _build_props_panel_chart(player_name: str, stat: str, n_games: int = 10):
    """Bar chart for the Best Props player panel. Last N games, no threshold line."""
    player_df = DF[DF["PLAYER_NAME"] == player_name].sort_values("_date", ascending=False).head(n_games)
    player_df = player_df.iloc[::-1]  # chronological

    fig = go.Figure()
    if player_df.empty or stat not in player_df.columns:
        fig.update_layout(template="plotly_dark", paper_bgcolor=COLORS["card"],
                          plot_bgcolor=COLORS["card"], height=260)
        return fig

    labels = [
        f"{r['_date'].strftime('%-m/%d') if pd.notna(r['_date']) else ''}<br>"
        + ("@ " if "@" in str(r.get("MATCHUP","")) else "vs ")
        + str(r.get("MATCHUP","")).split("@" if "@" in str(r.get("MATCHUP","")) else "vs.")[-1].strip()[:3]
        for _, r in player_df.iterrows()
    ]
    values = pd.to_numeric(player_df[stat], errors="coerce").fillna(0)
    avg_val = values.mean()

    stat_color = get_stat_color(stat)  # reuse existing helper
    bar_colors = [stat_color if v >= avg_val else "rgba(100,116,139,0.4)" for v in values]

    fig.add_trace(go.Bar(
        x=list(range(len(player_df))), y=values,
        marker_color=bar_colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=COLORS["text"]),
        hovertemplate=f"{stat}: %{{y}}<extra></extra>",
    ))
    fig.add_hline(y=avg_val, line_dash="dash", line_color=COLORS["text_secondary"],
                  line_width=1.5, annotation_text=f"Avg {avg_val:.1f}",
                  annotation_position="top left",
                  annotation_font_color=COLORS["text_secondary"])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        margin=dict(l=30,r=10,t=10,b=60), height=260, showlegend=False,
        xaxis=dict(tickvals=list(range(len(player_df))), ticktext=labels,
                   tickfont=dict(size=9), showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        bargap=0.25,
    )
    return fig
```

---

### Step 5 — New callback: render the panel
**File**: `dashboard/app.py`

```python
@callback(
    Output("props-player-panel", "children"),
    Output("props-player-panel", "style"),
    Input("props-panel-store", "data"),
    Input("props-panel-stat", "data"),
    prevent_initial_call=True,
)
def render_props_player_panel(store_data, active_stat_override):
    if not store_data:
        return None, {"width":"0px","minWidth":"0px","overflow":"hidden","transition":"width 0.3s ease"}

    player   = store_data["player"]
    def_stat = store_data.get("default_stat", "PTS")
    active   = active_stat_override or def_stat

    # Stat tabs: show the prop's default stat first, then the standard set
    PANEL_STATS = ["PTS","AST","REB","FG3M","STL","BLK"]
    if def_stat not in PANEL_STATS:
        PANEL_STATS = [def_stat] + PANEL_STATS  # put it first

    stat_tabs = []
    for s in PANEL_STATS:
        is_active = s == active
        stat_tabs.append(html.Div(
            s,
            id={"type": "props-panel-stat-tab", "index": s},
            n_clicks=0,
            className="tab active" if is_active else "tab",
            style={"fontSize":"0.75rem","padding":"4px 10px"},
        ))

    player_id  = PLAYER_IDS.get(player, "")
    photo_url  = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png" if player_id else ""
    chart_fig  = _build_props_panel_chart(player, active, n_games=10)

    panel_content = html.Div([
        # Header: photo + name + close button
        html.Div([
            html.Img(src=photo_url, style={"width":"44px","height":"44px","borderRadius":"50%",
                "objectFit":"cover","border":"2px solid rgba(20,184,166,0.4)"}) if photo_url else None,
            html.Div([
                html.Div(player, style={"fontWeight":"700","fontSize":"1rem","color":COLORS["text"]}),
                html.Div(f"Last 10 games", style={"fontSize":"0.78rem","color":COLORS["text_muted"]}),
            ], style={"flex":"1","marginLeft":"10px"}),
            # Close button
            html.Div("×", id="props-panel-close", n_clicks=0,
                     style={"cursor":"pointer","fontSize":"1.4rem","color":COLORS["text_muted"],
                            "padding":"0 4px","lineHeight":"1"}),
            # "Full analysis" link
            html.A("↗", href="/", id="props-panel-goto",
                   style={"color":"var(--accent-primary)","fontSize":"1.1rem","marginLeft":"8px",
                          "cursor":"pointer","textDecoration":"none"},
                   title="Open in Player Analysis"),
        ], style={"display":"flex","alignItems":"center","marginBottom":"12px"}),

        # Stat tabs
        html.Div(stat_tabs, className="tab-group",
                 style={"marginBottom":"10px","flexWrap":"wrap","gap":"4px"}),

        # Chart
        dcc.Graph(figure=chart_fig, config={"displayModeBar":False},
                  id="props-panel-chart"),

    ], style={"padding":"16px"}, className="card")

    panel_style = {
        "width": "420px", "minWidth": "420px",
        "overflow": "hidden",
        "transition": "width 0.3s ease, min-width 0.3s ease",
        "flexShrink": "0",
        "position": "sticky", "top": "80px",  # stays visible while scrolling
        "maxHeight": "calc(100vh - 100px)",
        "overflowY": "auto",
    }
    return panel_content, panel_style
```

---

### Step 6 — Stat tab click callback + close button
**File**: `dashboard/app.py`

```python
@callback(
    Output("props-panel-stat", "data"),
    Input({"type": "props-panel-stat-tab", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def switch_props_panel_stat(n_clicks_list):
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        raise PreventUpdate
    return triggered["index"]   # e.g. "PTS"

@callback(
    Output("props-panel-store", "data", allow_duplicate=True),
    Input("props-panel-close", "n_clicks"),
    prevent_initial_call=True,
)
def close_props_panel(n_clicks):
    if n_clicks:
        return None
    raise PreventUpdate
```

---

### Step 7 — "Go to full analysis" wires up player-dropdown
The `↗` link in the panel header navigates to `/` AND pre-selects the player:
```python
@callback(
    Output("player-dropdown", "value", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Input("props-panel-goto", "n_clicks"),
    State("props-panel-store", "data"),
    prevent_initial_call=True,
)
def goto_full_analysis(n_clicks, store_data):
    if n_clicks and store_data:
        return store_data["player"], "/"
    raise PreventUpdate
```

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/app.py:1664` | Modify | Wrap page in flex row, add `props-player-panel` div + 2 new stores |
| `dashboard/app.py:2195` | Modify | Wrap photo in clickable div, encode `player\|stat` in index |
| `dashboard/app.py:2213` | Modify | Encode `player\|stat` in player name span index |
| `dashboard/app.py:1989` | Modify | Remove navigation from name-click, replace with panel-store write |
| `dashboard/app.py:~2938` | Add | `_build_props_panel_chart()` helper function |
| `dashboard/app.py` | Add | `render_props_player_panel()` callback (panel renderer) |
| `dashboard/app.py` | Add | `switch_props_panel_stat()` callback (stat tab clicks) |
| `dashboard/app.py` | Add | `close_props_panel()` callback (close button) |
| `dashboard/app.py` | Add | `goto_full_analysis()` callback (↗ link) |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Panel breaks mobile layout | Use `display:none` on small screens via CSS media query |
| Circular callback (panel-store → panel render → stat tab → panel-store) | Use separate `props-panel-stat` store; only the stat tab writes to it |
| `allow_duplicate` on `props-panel-store` for close button | Use `allow_duplicate=True` on close callback output |
| Player name click currently navigates — breaking that could confuse users | Add clear `↗` icon for "full analysis" navigation so users have a path |
| `get_stat_color()` may not cover all stat labels (FG3M vs 3PM) | Add fallback color in `_build_props_panel_chart` |
| Panel stays open when switching stat filters | `props-panel-store` is a `dcc.Store` scoped to page — persists correctly |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (planned by Claude directly from codebase context)
- GEMINI_SESSION: N/A
