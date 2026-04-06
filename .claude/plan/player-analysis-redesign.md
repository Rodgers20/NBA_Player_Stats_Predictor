# Implementation Plan: Player Analysis Page Redesign

## Feature Summary
Redesign the Player Analysis page to match the reference screenshot:
- Large prominent search bar at the top
- Clean player header with bold name, team, position, and season averages inline
- Two-column layout: main chart (left) + Matchup Analysis panel (right)
- Bottom two-column: Season Trends (left) + Prop Bet History (right)
- Keep existing stat-type and time-period filter tabs at the top
- Dark solid card backgrounds (consistent with redesigned Best Props)
- Nav tabs right-aligned with white-pill active style

---

## Visual Diff (Current → Target)

| Element | Current | Target |
|---|---|---|
| Nav | Left logo + right links | Logo left + right tabs, active = white pill |
| Search | Dropdown 300px inside header card | Full-width search bar spanning whole page |
| Player header | Card with circle photo + dropdown | Clean text: "Name - TEAM • POS" + averages |
| Layout | Left chart (wide) + right sidebar | 2-col equal-ish: chart left, matchup right |
| Bottom | Supporting stats below chart | 2-col: Season Trends left, Prop History right |
| Cards | Glass/translucent | Solid dark #1c2131 |
| Filters | Teal active tab | White-pill active (same as Best Props) |

---

## Implementation Steps

### Step 1 — Nav: right-align tabs, white-pill active
**File**: `dashboard/app.py` — nav bar layout + CSS

Current nav has left logo + right links. Target has right-aligned pill tabs.

CSS changes in `custom.css`:
```css
/* Nav tabs white-pill (reuse .props-tabs pattern for all pages) */
.nav-tabs-group .nav-link {
    color: #6b7280;
    padding: 6px 18px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    border: none;
    background: transparent;
    transition: all 0.15s;
}
.nav-tabs-group .nav-link:hover { color: #d1d5db; background: rgba(255,255,255,0.07); }
.nav-tabs-group .nav-link.active {
    background: #ffffff;
    color: #0f1623;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
```

Nav layout change:
```python
# In create_app_layout() or wherever nav is built:
html.Div([
    html.Div("NBA Props AI", style={"fontWeight":"800","fontSize":"1.1rem","color":"#f0f4ff"}),
    html.Div([  # right side
        html.Div("Player Analysis", id="nav-player", className="nav-link active"),
        html.Div("Today's Games",   id="nav-games",  className="nav-link"),
        html.Div("Best Props",      id="nav-props",  className="nav-link"),
    ], className="nav-tabs-group", style={"display":"flex","gap":"4px","alignItems":"center"}),
], style={"display":"flex","justifyContent":"space-between","alignItems":"center",
          "padding":"0 32px","height":"56px","borderBottom":"1px solid rgba(255,255,255,0.06)"})
```

---

### Step 2 — Player search: full-width search bar
**File**: `dashboard/app.py` `create_player_analysis_page()` ~L643

Replace the header card (photo + 300px dropdown) with a wide search bar:

```python
# Search bar section (full width)
html.Div([
    html.Div([
        html.Span("🔍", style={"fontSize":"1.1rem","color":"#6b7280","marginRight":"12px","flexShrink":"0"}),
        dcc.Dropdown(
            id="player-dropdown",
            options=[{"label": p, "value": p} for p in PLAYERS],
            value=PLAYERS[0] if PLAYERS else None,
            placeholder="Search NBA Players...",
            searchable=True,
            clearable=False,
            className="player-search-fullwidth",
            style={"flex":"1","border":"none","background":"transparent"},
        ),
    ], style={
        "display":"flex","alignItems":"center","background":"#1c2131",
        "borderRadius":"12px","padding":"0 20px","height":"56px",
        "border":"1px solid rgba(255,255,255,0.1)",
        "boxShadow":"0 4px 20px rgba(0,0,0,0.3)",
    }),
], style={"marginBottom":"20px","maxWidth":"900px"}),
```

CSS for the fullwidth search dropdown:
```css
.player-search-fullwidth .Select-control {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1rem;
    color: #f0f4ff;
}
.player-search-fullwidth .Select-placeholder { color: #6b7280; font-size: 1rem; }
```

---

### Step 3 — Player header: name + stats inline (no photo)
**File**: `dashboard/app.py` — `update_player_header` callback output

Replace the current header display with:
```python
html.Div([
    # "LeBron James - LAL • SF/PF"
    html.Div([
        html.Span(player_name, style={"fontWeight":"800","fontSize":"1.6rem","color":"#f0f4ff"}),
        html.Span(f" - {team}", style={"fontWeight":"700","color":"#14b8a6","fontSize":"1.4rem"}),
        html.Span(f" • {position}", style={"fontWeight":"700","color":"#14b8a6","fontSize":"1.4rem"}),
    ], style={"marginBottom":"6px"}),
    # "Key season averages: 25.7 PPG, 7.3 RPG, ..."
    html.Div(
        f"Key season averages: {ppg} PPG, {rpg} RPG, {apg} APG, {fg_pct}% FG",
        style={"fontSize":"0.95rem","color":"#9ca3af"},
    ),
], style={"padding":"20px 0","marginBottom":"16px"})
```

The player photo div (`id="player-photo"`) can be hidden or removed — set `display:none` on the photo container in layout.

---

### Step 4 — Main two-column layout (chart + matchup side by side)
**File**: `dashboard/app.py` `create_player_analysis_page()` ~L680

Current: `html.Div([left_chart, right_sidebar], className="main-container")`

Change to explicit 2-col flex:
```python
html.Div([
    # LEFT: Stat filters + chart
    html.Div([
        # Stat tabs (keep existing, add props-tabs class for white-pill)
        html.Div([...STAT_TYPES tabs...], className="tab-group props-tabs"),
        # Time period tabs
        html.Div([L5, L10, L20, H2H, H/W, Season tabs], className="tab-group props-tabs"),
        # Main chart card
        html.Div([
            html.Div("Last 10 Games Performance vs. Betting Lines",
                     style={"fontWeight":"700","fontSize":"1rem","color":"#f0f4ff","marginBottom":"12px"}),
            dcc.Graph(id="main-chart", config={"displayModeBar":False}),
        ], style={"background":"#1c2131","borderRadius":"16px","padding":"20px",
                  "border":"1px solid rgba(255,255,255,0.07)"}),
    ], style={"flex":"2","minWidth":"0"}),

    # RIGHT: Matchup Analysis (sidebar-content → renamed to matchup panel)
    html.Div([
        html.Div("Matchup Analysis",
                 style={"fontWeight":"700","fontSize":"1rem","color":"#f0f4ff","marginBottom":"12px"}),
        html.Div(id="sidebar-content"),
    ], style={"flex":"1","minWidth":"280px","maxWidth":"380px",
              "background":"#1c2131","borderRadius":"16px","padding":"20px",
              "border":"1px solid rgba(255,255,255,0.07)"}),

], style={"display":"flex","gap":"16px","alignItems":"flex-start","marginBottom":"16px"}),
```

---

### Step 5 — Bottom two-column: Season Trends + Prop Bet History
**File**: `dashboard/app.py` — add below the main 2-col row

Currently "Supporting Stats" and the shooting breakdown chart live below the main chart. Restructure:

```python
html.Div([
    # LEFT: Season Trends (line chart — existing supporting stats chart or new)
    html.Div([
        html.Div("Season Trends",
                 style={"fontWeight":"700","fontSize":"1rem","color":"#f0f4ff","marginBottom":"4px"}),
        html.Div("Season averages vs. the most tip/low 10 the/ties average.",
                 style={"fontSize":"0.78rem","color":"#6b7280","marginBottom":"12px"}),
        dcc.Graph(id="shooting-breakdown-chart", config={"displayModeBar":False}),
    ], style={"flex":"1","background":"#1c2131","borderRadius":"16px","padding":"20px",
              "border":"1px solid rgba(255,255,255,0.07)"}),

    # RIGHT: Prop Bet History (sidebar-props content, existing bet history)
    html.Div([
        html.Div([
            html.Div("Prop Bet History",
                     style={"fontWeight":"700","fontSize":"1rem","color":"#f0f4ff"}),
            html.Div("Recent Game", style={
                "fontSize":"0.75rem","color":"#6b7280",
                "border":"1px solid rgba(255,255,255,0.1)",
                "borderRadius":"6px","padding":"3px 10px","cursor":"pointer"
            }),
        ], style={"display":"flex","justifyContent":"space-between",
                  "alignItems":"center","marginBottom":"12px"}),
        html.Div(id="prop-bet-history-panel"),
    ], style={"flex":"1","background":"#1c2131","borderRadius":"16px","padding":"20px",
              "border":"1px solid rgba(255,255,255,0.07)"}),

], style={"display":"flex","gap":"16px","marginBottom":"16px"}),
```

The `prop-bet-history-panel` is populated by a new callback that reads the sidebar-props content (already computed in `render_sidebar_props`).

---

### Step 6 — CSS: card backgrounds + filter tab inheritance
**File**: `dashboard/assets/custom.css`

```css
/* Solid dark card override for new layout areas */
.analysis-card {
    background: #1c2131;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.07);
}
/* Player analysis filter tabs: white pill (inherit .props-tabs) */
/* Already covered by .props-tabs CSS added in Best Props redesign */
```

---

### Step 7 — Threshold slider + hit-rate header: keep but style consistently
The threshold slider and hit-rate header stay in the left column above the chart. Style:
- Wrap in the same `analysis-card` div or inline solid dark container
- Keep all existing callback logic untouched

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/assets/custom.css` | Add | `.nav-tabs-group`, `.player-search-fullwidth`, `.analysis-card` |
| `dashboard/app.py` nav section | Modify | Nav: right-align + white-pill active |
| `dashboard/app.py:643` | Modify | Replace search header card with full-width search bar |
| `dashboard/app.py:650` | Modify | Hide player-photo div (set display:none) |
| `dashboard/app.py:674` | Modify | Player header callback output — new "Name - TEAM • POS" format |
| `dashboard/app.py:680` | Modify | Main layout: left chart + right matchup panel side by side |
| `dashboard/app.py:840` | Modify | Bottom 2-col: Season Trends + Prop Bet History |
| `dashboard/app.py:684` | Modify | Add `props-tabs` class to stat and period tab groups |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `update_player_header` callback needs PPG/RPG/APG | Already computed in `render_player_header`; extract and expose |
| Sidebar callbacks (`sidebar-matchup`, etc.) still expect old sidebar layout | Sidebar content still renders into `sidebar-content` div — just moves visually |
| Nav link active class managed by `display_page` callback | Keep existing `Output("nav-player","className")` etc. — just update CSS class names |
| `player-photo` div removal breaks `update_player_photo` callback | Set `display:none` on container, keep the div/callback intact |
| Bottom prop-bet-history needs data from sidebar-props | Add `Output("prop-bet-history-panel","children")` to existing `render_sidebar_content` callback |
| Threshold slider moving to new column may need ID reference update | No ID changes — just DOM position changes, Dash callbacks are ID-based |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
