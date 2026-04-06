# Implementation Plan: Player Analysis Page v2

## Feature Summary
Redesign the Player Analysis page to match the reference screenshots:

**Screenshot 2 (full page):**
- Player header: circular photo + bold name + teal team badge pill + amber injury badge + stat chips (26.1 PTS · 10.5 AST · 12.6 REB)
- Right sidebar: 3 always-visible stacked sections — Matchup Analysis + Injury Context + Prop Bet History (NO tab switching)
- Season Trends chart at bottom spanning full width below main 2-col

**Screenshot 1 (Matchup Analysis card zoom):**
- "Matchup Analysis" as large card title
- "Key {OPP} Defense vs {Position}  {W}-{L}" bold header
- Sub-line: "Opp FG% X.X% (#N)  TOV forced X.X/g (MN)"
- Underline-style tabs: Overall | vs Guards | vs Forwards | vs Centers
- Stats table with **amber** rank badges (#11, #13, #18) + value column
- Polar radar chart: green fill = Strengths (low rank = easy defense), red fill = Weaknesses (high rank = hard defense), labels: Strengths / Defensive Strengths / Defensive Strompts / Weaknesses

---

## Visual Diff (Current → Target)

| Element | Current | Target |
|---|---|---|
| Player header | Text-only name + averages | Circular photo + name + teal team pill + amber injury pill + stat chips |
| Right sidebar | 4 tab buttons (Matchup/Injuries/Insights/Props) | Always-visible 3 stacked cards (no tabs) |
| Matchup card | CARD style, orange rank colors | Redesigned: amber ranks, underline tabs, polar radar chart |
| Injury section | Separate tab in sidebar | "Injury Context" card, always visible below Matchup |
| Prop Bet History | Separate tab in sidebar | Always-visible card below Injury Context |
| Season Trends | Bottom-right of 2-col row | Full-width card below the 2-col row |

---

## Implementation Steps

### Step 1 — CSS additions
**File**: `dashboard/assets/custom.css`

Add after the existing `.analysis-filters` section:

```css
/* ── Player header v2 ── */
.player-header-v2 {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 20px 0;
    margin-bottom: 16px;
}
.player-photo-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    object-fit: cover;
    object-position: top center;
    border: 2px solid rgba(20,184,166,0.4);
    flex-shrink: 0;
    background: rgba(20,184,166,0.08);
}
.player-photo-initial-circle {
    width: 80px; height: 80px; border-radius: 50%;
    background: rgba(20,184,166,0.12);
    border: 2px solid rgba(20,184,166,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; font-weight: 800; color: #14b8a6;
    flex-shrink: 0;
}
.team-badge {
    display: inline-flex; align-items: center;
    padding: 3px 12px; border-radius: 9999px;
    background: rgba(20,184,166,0.15);
    border: 1px solid rgba(20,184,166,0.35);
    color: #2dd4bf; font-size: 0.82rem; font-weight: 700;
    letter-spacing: 0.04em;
}
.injury-badge {
    display: inline-flex; align-items: center;
    padding: 3px 12px; border-radius: 9999px;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.02em;
}
.injury-badge-gtd    { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #f59e0b; }
.injury-badge-out    { background: rgba(244,63,94,0.15);  border: 1px solid rgba(244,63,94,0.4);  color: #f43f5e; }
.injury-badge-active { background: rgba(34,197,94,0.12);  border: 1px solid rgba(34,197,94,0.3);  color: #22c55e; }

.player-stat-chips {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px;
}
.player-stat-chip {
    font-size: 0.82rem; font-weight: 700; color: #d1d5db;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 6px; padding: 3px 10px;
}
.player-stat-chip span.chip-label { color: #6b7280; font-weight: 400; margin-left: 4px; }

/* ── Matchup card v2 ── */
.matchup-card-v2 { }   /* inherits .analysis-card */

.matchup-section-title {
    font-size: 1.1rem; font-weight: 800; color: #f0f4ff;
    letter-spacing: -0.01em; margin-bottom: 14px;
}
.matchup-defense-header {
    font-size: 0.92rem; font-weight: 700; color: #f0f4ff;
    margin-bottom: 4px;
}
.matchup-defense-sub {
    font-size: 0.75rem; color: #6b7280; margin-bottom: 14px;
}
.matchup-record { color: #6b7280; font-weight: 500; margin-left: 8px; font-size: 0.82rem; }

/* Underline tabs for matchup (not pill style) */
.matchup-tabs {
    display: flex; gap: 0; border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
}
.matchup-tab {
    padding: 6px 14px; font-size: 0.82rem; font-weight: 500;
    color: #6b7280; cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s, border-color 0.15s;
    white-space: nowrap;
}
.matchup-tab.active {
    color: #f0f4ff; border-bottom-color: #f0f4ff;
}

/* Matchup stats table */
.matchup-stats-table { width: 100%; border-collapse: collapse; }
.matchup-stats-table tr td { padding: 8px 0; }
.matchup-stats-table tr + tr td { border-top: 1px solid rgba(255,255,255,0.04); }
.matchup-stat-label { font-size: 0.88rem; font-weight: 500; color: #f0f4ff; }
.matchup-stat-header { font-size: 0.75rem; color: #6b7280; font-weight: 400; }
.matchup-rank { font-size: 0.88rem; font-weight: 700; color: #f59e0b; text-align: center; }
.matchup-stat-val { font-size: 0.95rem; font-weight: 700; color: #f0f4ff; text-align: right; }

/* Polar radar container */
.matchup-radar-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px; margin-top: 16px;
}
.matchup-radar-title {
    font-size: 0.82rem; font-weight: 600; color: #9ca3af;
    text-align: center; margin-bottom: 2px;
}

/* Injury context compact */
.injury-context-card { }  /* inherits .analysis-card */
.injury-context-status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 8px; font-size: 0.82rem; font-weight: 600;
    margin-bottom: 10px;
}
.injury-context-body {
    font-size: 0.82rem; color: #d1d5db; line-height: 1.6;
}
.injury-impact-note {
    font-size: 0.78rem; color: #9ca3af; margin-top: 8px; line-height: 1.5;
}

/* Prop bet history compact */
.prop-bet-row {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 0; font-size: 0.78rem; color: #d1d5db;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.prop-bet-row:last-child { border-bottom: none; }
.prop-bet-opponent { color: #6b7280; min-width: 60px; }
.prop-bet-hit  { color: #22c55e; font-weight: 700; }
.prop-bet-miss { color: #f43f5e; font-weight: 700; }
```

---

### Step 2 — Update `update_player_header()` callback
**File**: `dashboard/app.py` ~L2290

Replace the return value to include circular photo + badges + stat chips:

```python
# Get injury status for badge
try:
    inj = get_player_injury_status(player_name)
    inj_status = inj.get("status", "ACTIVE")
    inj_reason = inj.get("reason", "")
except Exception:
    inj_status = "ACTIVE"
    inj_reason = ""

# Map status → badge class + label
BADGE_MAP = {
    "ACTIVE":       ("injury-badge injury-badge-active", "Active"),
    "QUESTIONABLE": ("injury-badge injury-badge-gtd",    "Questionable"),
    "DOUBTFUL":     ("injury-badge injury-badge-gtd",    "Doubtful"),
    "GTD":          ("injury-badge injury-badge-gtd",    "GTD - Game Time Decision"),
    "OUT":          ("injury-badge injury-badge-out",    "OUT"),
    "PROBABLE":     ("injury-badge injury-badge-active", "Probable"),
}
badge_cls, badge_label = BADGE_MAP.get(inj_status, ("injury-badge injury-badge-active", inj_status))
if inj_reason and inj_status not in ("ACTIVE", "UNKNOWN"):
    badge_label = f"{inj_status.title()} - {inj_reason}"

photo_url = get_player_headshot_url(player_name)

return html.Div([
    # Photo (circular)
    html.Img(src=photo_url, className="player-photo-circle")
    if photo_url else html.Div(player_name[0].upper(), className="player-photo-initial-circle"),

    # Name + badges + stat chips
    html.Div([
        # Row 1: name + team badge + injury badge
        html.Div([
            html.Span(player_name, style={"fontWeight":"800","fontSize":"1.5rem",
                                          "color":"#f0f4ff","letterSpacing":"-0.02em","marginRight":"10px"}),
            html.Span(team, className="team-badge") if team else None,
            html.Span(badge_label, className=badge_cls, style={"marginLeft":"8px"}),
        ], style={"display":"flex","alignItems":"center","flexWrap":"wrap","marginBottom":"6px"}),

        # Row 2: stat chips
        html.Div([
            html.Div([html.Span(f"{avg_pts:.1f}"), html.Span("PTS", className="chip-label")], className="player-stat-chip"),
            html.Div([html.Span(f"{avg_ast:.1f}"), html.Span("AST", className="chip-label")], className="player-stat-chip"),
            html.Div([html.Span(f"{avg_reb:.1f}"), html.Span("REB", className="chip-label")], className="player-stat-chip"),
        ] + ([html.Div([html.Span(f"{avg_fg:.1f}%"), html.Span("FG", className="chip-label")], className="player-stat-chip")] if avg_fg else []),
        className="player-stat-chips"),
    ]),
], className="player-header-v2")
```

---

### Step 3 — Remove sidebar tab buttons, restructure right column
**File**: `dashboard/app.py` `create_player_analysis_page()` ~L750–759

Replace:
```python
# RIGHT — matchup / injuries / insights / props sidebar
html.Div([
    html.Div([
        html.Button("Matchup",    id="sidebar-matchup",   n_clicks=1, className="tab active"),
        html.Button("Injuries",   id="sidebar-injuries",  n_clicks=0, className="tab"),
        html.Button("Insights",   id="sidebar-insights",  n_clicks=0, className="tab"),
        html.Button("Best Props", id="sidebar-props",     n_clicks=0, className="tab"),
    ], className="tab-group", style={"marginBottom": "12px"}),
    html.Div(id="sidebar-content"),
], className="analysis-side-col analysis-card"),
```

With (remove tab buttons, keep sidebar-content, add stub IDs for dead callbacks):
```python
# RIGHT — always-visible 3-section sidebar
html.Div([
    # Hidden stubs to keep dead tab-button callbacks from crashing
    html.Div(id="sidebar-matchup",  style={"display":"none"}, **{"n_clicks":0}),
    html.Div(id="sidebar-injuries", style={"display":"none"}, **{"n_clicks":0}),
    html.Div(id="sidebar-insights", style={"display":"none"}, **{"n_clicks":0}),
    html.Div(id="sidebar-props",    style={"display":"none"}, **{"n_clicks":0}),
    html.Div(id="sidebar-content"),
], className="analysis-side-col"),
```

---

### Step 4 — `update_sidebar_content` callback: always return all 3 sections
**File**: `dashboard/app.py` ~L2767

Change callback to always render all 3 sections stacked (ignore tab store):

```python
@callback(
    Output("sidebar-content", "children"),
    [Input("player-dropdown", "value"),
     Input("selected-stat", "data")]
)
def update_sidebar_content(player_name, stat):
    if not player_name:
        return None
    return html.Div([
        create_matchup_content_v2(player_name, stat),    # Matchup Analysis card
        create_injury_context_card(player_name),          # Injury Context card
        create_prop_bet_history_card(player_name, stat),  # Prop Bet History card
    ])
```

**Keep old `update_sidebar_tabs` callback** but make it a no-op (it still outputs className for the 4 hidden divs — just return ["", "", "", ""] always).

---

### Step 5 — Redesign `create_matchup_content()` → `create_matchup_content_v2()`
**File**: `dashboard/app.py` ~L3554

Keep existing data-fetch logic (opponent, position, defense stats) intact. Replace the HTML structure:

```python
def create_matchup_content_v2(player_name, stat):
    # ... (keep all existing data fetching: opponent, opp_record, pts/ast/reb/tpm allowed+ranks)

    # ── Polar radar chart ─────────────────────────────────────────────────
    # Normalize ranks 1-30: lower rank = easier defense for player = higher score
    def norm(rank): return max(0, (30 - rank) / 29)

    categories  = ["Pts", "Ast", "Reb", "3PM"]
    scores      = [norm(pts_rank), norm(ast_rank), norm(reb_rank), norm(tpm_rank)]
    avg_score   = sum(scores) / 4

    green_vals = [v if v >= 0.5 else 0 for v in scores]
    red_vals   = [1 - v if v < 0.5 else 0 for v in scores]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=green_vals + [green_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(34,197,94,0.25)',
        line=dict(color='rgba(34,197,94,0.7)', width=1.5),
        name='Favorable', showlegend=False,
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=red_vals + [red_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(244,63,94,0.25)',
        line=dict(color='rgba(244,63,94,0.7)', width=1.5),
        name='Unfavorable', showlegend=False,
    ))
    radar_fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=False, range=[0,1]),
            angularaxis=dict(tickfont=dict(size=10, color='#9ca3af'), gridcolor='rgba(255,255,255,0.08)'),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=30, b=30), height=200,
        showlegend=False,
    )

    # ── Underline-style position tabs ────────────────────────────────────
    position_tabs = ["Overall", "vs Guards", "vs Forwards", "vs Centers"]
    active_tab = f"vs {player_pos_display}"

    return html.Div([
        # Title
        html.Div("Matchup Analysis", className="matchup-section-title"),

        # Defense header + record
        html.Div([
            html.Span(f"Key {opponent} Defense vs {player_pos_display}",
                      className="matchup-defense-header"),
            html.Span(f"  {opp_record}", className="matchup-record") if opp_record else None,
        ], style={"marginBottom":"4px"}),

        # Sub-line
        html.Div([
            html.Span(f"Opp FG% {opp_fg_pct*100:.1f}% (#{int(opp_fg_rank)})  " if opp_fg_pct else "",
                      style={"marginRight":"12px"}),
            html.Span(f"TOV forced {opp_tov:.1f}/g (M{int(opp_tov_rank)})" if opp_tov else ""),
        ], className="matchup-defense-sub"),

        # Underline tabs
        html.Div([
            html.Span(pos, className=f"matchup-tab {'active' if pos == active_tab or (pos == 'Overall' and active_tab not in position_tabs) else ''}")
            for pos in position_tabs
        ], className="matchup-tabs"),

        # Stats table
        html.Table([
            html.Thead(html.Tr([
                html.Td("Stat (per game)", className="matchup-stat-header"),
                html.Td("Rank",            className="matchup-stat-header", style={"textAlign":"center"}),
                html.Td("Avg",             className="matchup-stat-header", style={"textAlign":"right"}),
            ])),
            html.Tbody([
                html.Tr([html.Td("Points Allowed",   className="matchup-stat-label"),
                         html.Td(f"#{pts_rank}",     className="matchup-rank"),
                         html.Td(f"{pts_allowed:.1f}", className="matchup-stat-val")]),
                html.Tr([html.Td("Assists Allowed",  className="matchup-stat-label"),
                         html.Td(f"#{ast_rank}",     className="matchup-rank"),
                         html.Td(f"{ast_allowed:.1f}", className="matchup-stat-val")]),
                html.Tr([html.Td("Rebounds Allowed", className="matchup-stat-label"),
                         html.Td(f"#{reb_rank}",     className="matchup-rank"),
                         html.Td(f"{reb_allowed:.1f}", className="matchup-stat-val")]),
                html.Tr([html.Td("3-Pointers Allowed", className="matchup-stat-label"),
                         html.Td(f"#{tpm_rank}",     className="matchup-rank"),
                         html.Td(f"{tpm_allowed:.1f}", className="matchup-stat-val")]),
            ])
        ], className="matchup-stats-table"),

        # Radar chart
        html.Div([
            html.Div(f"{opponent} Defense Rating #{pts_rank} vs. {player_pos_display}",
                     className="matchup-radar-title"),
            html.Div([
                html.Span("Strengths", style={"fontSize":"0.7rem","color":"#22c55e","marginRight":"16px"}),
                html.Span("Weaknesses", style={"fontSize":"0.7rem","color":"#f43f5e"}),
            ], style={"textAlign":"center","marginBottom":"4px"}),
            dcc.Graph(figure=radar_fig, config={"displayModeBar":False}),
        ], className="matchup-radar-container"),

    ], className="analysis-card matchup-card-v2")
```

---

### Step 6 — New `create_injury_context_card()` helper
**File**: `dashboard/app.py` (add near `create_injuries_content`)

Compact version for the always-visible sidebar:

```python
def create_injury_context_card(player_name):
    try:
        status = get_player_injury_status(player_name)
        status_text = status.get("status", "ACTIVE")
        reason = status.get("reason", "")
        news = status.get("news", [])
    except Exception:
        status_text, reason, news = "UNKNOWN", "", []

    STATUS_STYLES = {
        "ACTIVE":       ("#22c55e", "rgba(34,197,94,0.12)"),
        "QUESTIONABLE": ("#f59e0b", "rgba(245,158,11,0.12)"),
        "DOUBTFUL":     ("#f97316", "rgba(249,115,22,0.12)"),
        "OUT":          ("#f43f5e", "rgba(244,63,94,0.12)"),
        "PROBABLE":     ("#a3e635", "rgba(163,230,53,0.12)"),
        "UNKNOWN":      ("#64748b", "rgba(100,116,139,0.12)"),
    }
    color, bg = STATUS_STYLES.get(status_text, STATUS_STYLES["ACTIVE"])

    news_items = [n.get("title","") for n in news[:2] if n.get("title")]
    impact = news_items[0] if news_items else ""

    return html.Div([
        html.Div("Injury Context", className="matchup-section-title"),
        html.Div([
            html.Span("Status: ", style={"color":"#9ca3af"}),
            html.Span(f"{status_text.title()}" + (f" ({reason})" if reason else ""),
                      style={"color": color, "fontWeight":"600"}),
        ], style={"fontSize":"0.85rem","marginBottom":"8px"}),
        html.Div(impact, className="injury-context-body") if impact else
        html.Div("No recent injury reports.", className="injury-context-body",
                 style={"color":"#6b7280"}),
        html.Div(
            "Impact: Minutes may be limited if active. Consider lower prop lines.",
            className="injury-impact-note"
        ) if status_text in ("QUESTIONABLE","DOUBTFUL","GTD") else None,
    ], className="analysis-card injury-context-card")
```

---

### Step 7 — New `create_prop_bet_history_card()` helper
**File**: `dashboard/app.py` (add near `create_best_props_content`)

Shows last 5 prop bets for the player from the existing props cache:

```python
def create_prop_bet_history_card(player_name, stat):
    from utils.props_cache import get_cached_props
    try:
        cache = get_cached_props()
        all_props = cache.get("main_page_data", [])
    except Exception:
        all_props = []

    # Find this player's prop data
    player_props = [p for p in all_props if p.get("player","") == player_name]

    rows = []
    for p in player_props[:5]:
        opp = p.get("opponent","")
        direction = p.get("direction","Over")
        line = p.get("line","")
        hit_rate = p.get("hit_rate",0)
        hits = p.get("hits",0)
        total = p.get("total",0)
        stat_label = p.get("stat","PTS")
        hit_pct = int(hit_rate * 100)
        hit_cls = "prop-bet-hit" if hit_pct >= 65 else "prop-bet-miss"
        rows.append(html.Div([
            html.Span(f"vs. {opp}", className="prop-bet-opponent"),
            html.Span(f"{direction} {line} {stat_label}",
                      style={"flex":"1","color":"#d1d5db"}),
            html.Span(f"{hits}/{total} ({hit_pct}%)", className=hit_cls),
        ], className="prop-bet-row"))

    if not rows:
        rows = [html.Div("No prop data available.",
                         style={"fontSize":"0.8rem","color":"#6b7280","padding":"8px 0"})]

    return html.Div([
        html.Div([
            html.Span("Prop Bet History", className="matchup-section-title",
                      style={"marginBottom":"0"}),
            html.Span("Recent", style={"fontSize":"0.72rem","color":"#6b7280",
                                        "border":"1px solid rgba(255,255,255,0.1)",
                                        "borderRadius":"6px","padding":"2px 8px"}),
        ], style={"display":"flex","justifyContent":"space-between",
                  "alignItems":"center","marginBottom":"12px"}),
        html.Div(rows),
    ], className="analysis-card")
```

---

### Step 8 — Fix `update_sidebar_tabs` callback to not crash
**File**: `dashboard/app.py` ~L1846

The existing callback outputs classNames for the 4 sidebar tab buttons. Since we're keeping the div IDs but hiding them, the callback still needs to exist. Change it to simply return 4 empty strings or keep it as-is (it still works on hidden divs, no visual impact).

---

### Step 9 — Season Trends: move to full-width below the 2-col row
**File**: `dashboard/app.py` `create_player_analysis_page()` ~L820

Currently Season Trends is in `.analysis-bottom-row` right column. Since the right sidebar now always shows Matchup+Injury+Props, move Season Trends to a full-width card below the bottom row:

```python
# AFTER the analysis-bottom-row div:
html.Div([
    html.Div("Season Trends", className="analysis-card-title"),
    html.Div("Season averages vs. the most recent 10-game average.",
             style={"fontSize":"0.78rem","color":"#6b7280","marginBottom":"14px"}),
    dcc.Graph(id="season-trends-chart", config={"displayModeBar":False},
              style={"height":"280px"}),
], className="analysis-card"),
```

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/assets/custom.css` | Add | `.player-header-v2`, `.player-photo-circle`, `.team-badge`, `.injury-badge-*`, `.player-stat-chip*`, `.matchup-*` classes |
| `dashboard/app.py:2290` | Modify | `update_player_header()` — photo + badges + stat chips |
| `dashboard/app.py:750` | Modify | Right sidebar: remove tab buttons, keep sidebar-content div |
| `dashboard/app.py:2767` | Modify | `update_sidebar_content` — ignore tab, always return 3 stacked sections |
| `dashboard/app.py:3554` | Modify | `create_matchup_content()` → add v2 design: amber ranks, underline tabs, polar radar |
| `dashboard/app.py:3836` | Add near | `create_injury_context_card()` — compact injury for always-visible sidebar |
| `dashboard/app.py:4414` | Add near | `create_prop_bet_history_card()` — compact prop history card |
| `dashboard/app.py:820` | Modify | Season Trends: move to full-width card below bottom row |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `update_sidebar_tabs` callback still outputs to `sidebar-matchup` etc. IDs | Keep div IDs in layout, just hide them — callback still fires, no crash |
| `sidebar-tab` store still exists and old callbacks read from it | Change `update_sidebar_content` to not use tab store; keep store for compat |
| Polar chart looks wrong for players with no defense data (all zeros) | Guard: if all ranks == 15 (default), show a message instead of chart |
| `get_player_injury_status` may be slow (network call) | Already cached in `injury_news.py`; sidebar renders on player change anyway |
| Removing Season Trends from bottom-right leaves bottom row with only left col | Wrap left col in `analysis-bottom-row` with `flex: 1` then Season Trends full-width below |

---

## SESSION_ID
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
