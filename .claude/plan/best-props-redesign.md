# Implementation Plan: Best Props Page Redesign

## Feature Summary
Redesign the Best Props page to match the reference screenshot:
- 4-column card grid (not a narrow single-column list)
- Cards show large rectangular player photo + floating team logo
- Primary metric: big bold "Win probability XX% EV ↑" (not hit rate)
- AI Insight section with one narrative + single bottom tag
- Plain white title "Today's Top Value Props" (no gradient)
- Tabs: white-filled active pill on transparent bar

---

## Visual Diff (Current → Target)

| Element | Current | Target |
|---|---|---|
| Layout | Single column, maxWidth 800px | 4-col responsive grid, max 1400px |
| Card photo | 52px circle, left-aligned | Full-width rectangular crop, ~140px tall, top of card |
| Team logo | None on card | Small logo floating top-right of card |
| Primary metric | Hit% large (1.5rem) | EV% large (2rem+) labeled "Win probability" |
| Insight | Bullet factors list + narrative | "AI Insight" label + 1 text paragraph + 1 bottom tag |
| Title | Gradient "BEST PROPS" | Plain white "Today's Top Value Props" |
| Tabs (active) | Teal gradient fill | White/light fill, dark text |
| Tab bar | Background container with border | Transparent, pills float freely |

---

## Implementation Steps

### Step 1 — CSS: New prop card grid + card visual structure
**File**: `dashboard/assets/custom.css`

Add/replace `.prop-card` and add new helper classes:

```css
/* Props page grid container */
.props-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
}
@media (max-width: 1100px) { .props-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 780px)  { .props-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 500px)  { .props-grid { grid-template-columns: 1fr; } }

/* Redesigned prop card */
.prop-card-v2 {
    position: relative;
    background: #1c2131;          /* solid dark, not glass */
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.07);
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
}
.prop-card-v2:hover {
    transform: translateY(-4px);
    border-color: rgba(255,255,255,0.18);
    box-shadow: 0 12px 32px rgba(0,0,0,0.4);
}

/* Player photo banner */
.prop-card-photo-wrap {
    position: relative;
    width: 100%;
    height: 140px;
    overflow: hidden;
    background: #161b2a;
    flex-shrink: 0;
}
.prop-card-photo-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: top center;
}
/* Team logo overlay */
.prop-card-team-logo {
    position: absolute;
    top: 10px;
    right: 10px;
    width: 38px;
    height: 38px;
    object-fit: contain;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
}

/* Card body */
.prop-card-body {
    padding: 14px 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 0;
    flex: 1;
}

/* EV colored classes */
.ev-green  { color: #22c55e; }
.ev-orange { color: #f97316; }
.ev-red    { color: #ef4444; }

/* Bottom tag */
.prop-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-top: auto;
    padding-top: 10px;
}
.prop-tag-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Active tab override for props page — white pill */
.props-tabs .tab.active {
    background: #ffffff;
    color: #0f1623;
    border: none;
    box-shadow: none;
}
.props-tabs .tab {
    color: #6b7280;
    background: transparent;
    border: none;
}
.props-tabs .tab:hover {
    color: #d1d5db;
    background: rgba(255,255,255,0.06);
}
.props-tabs .tab-group {
    background: transparent;
    border: none;
    padding: 0;
    gap: 4px;
}
```

---

### Step 2 — Page layout update: `create_best_props_page()`
**File**: `dashboard/app.py` ~L1664

Changes:
1. Title: replace gradient "BEST PROPS" with plain `"Today's Top Value Props"` in white
2. Remove maxWidth 800px from right-column div → let it fill full width
3. Add `className="props-tabs"` to the stat filter tab-group div
4. Change `props-list` container: add `className="props-grid"` (grid applied via CSS)
5. Expand outer maxWidth from 1280px → 1400px

```python
# Title change (L1703-1714):
html.Div("Today's Top Value Props", style={
    "fontSize": "1.8rem", "fontWeight": "800",
    "color": "#f0f4ff", "marginBottom": "6px",
    "letterSpacing": "-0.02em",
}),

# Tab group — add className="props-tabs":
], className="tab-group props-tabs", style={"marginBottom": "12px", "flexWrap": "wrap"}),

# Right column — remove maxWidth: 800px:
], style={"flex": "1", "minWidth": "0"}),   # was "maxWidth": "800px"

# Outer container — widen:
], style={"display":"flex","gap":"20px","alignItems":"flex-start","maxWidth":"1400px","margin":"0 auto"}),
```

---

### Step 3 — Rebuild prop card HTML
**File**: `dashboard/app.py` prop card builder ~L2243

Replace current `prop_card = html.Div([...], className="card prop-card")` with new structure:

```python
# EV color logic (replaces hit_color as primary):
ev_class = "ev-green" if hit_pct >= 70 else "ev-orange" if hit_pct >= 60 else "ev-red"
ev_arrow = " ↑" if trend == "positive" or hit_pct >= 70 else ""

# Tag logic (pick best factor or fallback):
if factors:
    best_factor = factors[0]
    tag_positive = best_factor.get("positive", True)
    tag_text = best_factor.get("text", "")
else:
    tag_positive = hit_pct >= 65
    tag_text = f"{hit_pct}% hit rate ({location_label})"
tag_dot_color = "#22c55e" if tag_positive is True else "#f97316" if tag_positive is False else "#6b7280"

# Stat line label mapping:
STAT_LABELS = {"PTS":"Points","AST":"Assists","REB":"Rebounds","FG3M":"3-Pointers","STL":"Steals","BLK":"Blocks"}
stat_label = STAT_LABELS.get(prop.get("stat","PTS"), prop.get("stat",""))
direction = prop.get("direction", "Over")
line = prop.get("line", "")
stat_line_text = f"{stat_label}: {direction} {line}" if line else stat_label

# Team logo:
team_logo = get_team_logo_url(prop.get("team", ""))

# Card:
prop_card = html.Div([
    # Photo banner + team logo overlay
    html.Div([
        html.Img(
            src=player_photo, alt=prop.get("player",""),
            style={"width":"100%","height":"100%","objectFit":"cover","objectPosition":"top center"}
        ) if player_photo else html.Div(style={"width":"100%","height":"100%","background":"#161b2a"}),
        html.Img(src=team_logo, className="prop-card-team-logo") if team_logo else None,
    ], className="prop-card-photo-wrap",
       id={"type":"prop-player-photo","index":f"{prop.get('player','')}|{prop.get('stat','PTS')}"},
       n_clicks=0,
    ),

    # Card body
    html.Div([
        # Player name + team/position
        html.Div(
            prop.get("player",""),
            id={"type":"prop-player-name","index":f"{prop.get('player','')}|{prop.get('stat','PTS')}"},
            n_clicks=0,
            style={"fontWeight":"800","fontSize":"1.1rem","color":"#f0f4ff","cursor":"pointer","marginBottom":"2px"},
        ),
        html.Div(
            f"{prop.get('team','')}{',' if prop.get('position') else ''} {prop.get('position','')}",
            style={"fontSize":"0.82rem","color":"#6b7280","marginBottom":"10px"}
        ),

        # Stat line (e.g. "Points: Over 12.5")
        html.Div([
            html.Span(f"{stat_label}: ", style={"fontWeight":"400","color":"#d1d5db"}),
            html.Span(f"{direction} {line}", style={"fontWeight":"700","color":"#f0f4ff"}),
        ], style={"fontSize":"0.95rem","marginBottom":"8px"}),

        # Divider
        html.Hr(style={"border":"none","borderTop":"1px solid rgba(255,255,255,0.07)","margin":"8px 0"}),

        # Win probability label
        html.Div("Win probability", style={"fontSize":"0.78rem","color":"#6b7280","marginBottom":"4px"}),

        # Big EV number
        html.Div(
            f"{hit_pct}% EV{ev_arrow}",
            className=ev_class,
            style={"fontSize":"2rem","fontWeight":"800","lineHeight":"1","marginBottom":"12px"},
        ),

        # AI Insight
        html.Div("AI Insight", style={"fontSize":"0.78rem","fontWeight":"700","color":"#6b7280","marginBottom":"4px"}),
        html.Div(
            narrative or f"{location_label}: {hits}/{total} games hit ({hit_pct}%)",
            style={"fontSize":"0.82rem","color":"#d1d5db","lineHeight":"1.5","marginBottom":"8px"},
        ),

        # Bottom tag
        html.Div([
            html.Div(className="prop-tag-dot", style={"backgroundColor": tag_dot_color}),
            html.Span(tag_text, style={"color": tag_dot_color if tag_positive is not None else "#6b7280"}),
        ], className="prop-tag"),

    ], className="prop-card-body"),
], className="prop-card-v2")
```

---

### Step 4 — Change `props-list` to use grid class
**File**: `dashboard/app.py` ~L1775 and the return of `update_best_props_main`

The `html.Div(id="props-list")` renders prop_cards as children. The grid is applied via CSS on `.props-grid`, so we need to either:
- Add `className="props-grid"` to the `html.Div(id="props-list")` wrapper, OR
- Return cards wrapped in `html.Div(cards, className="props-grid")` from the callback

**Preferred**: wrap in callback return so it's always applied:
```python
# In update_best_props_main callback return (last line):
return [html.Div(prop_cards, className="props-grid")]
```

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/assets/custom.css` | Add | New `.prop-card-v2`, `.props-grid`, `.props-tabs`, `.prop-tag`, EV color classes |
| `dashboard/app.py:1703` | Modify | Title: remove gradient, set plain white "Today's Top Value Props" |
| `dashboard/app.py:1719` | Modify | Add `className="props-tabs"` to stat filter tab-group |
| `dashboard/app.py:1779` | Modify | Remove `maxWidth: 800px` from right column |
| `dashboard/app.py:1781` | Modify | Expand outer maxWidth to 1400px |
| `dashboard/app.py:2243` | Modify | Rebuild card HTML to `prop-card-v2` structure |
| `dashboard/app.py:~2330` | Modify | Wrap return in `html.Div(prop_cards, className="props-grid")` |

---

## Risks and Mitigation

| Risk | Mitigation |
|---|---|
| `direction` / `line` not in prop dict | Fallback to `stat_label` only; check existing prop dict keys first |
| Photo aspect ratio varies — cropped faces | Use `object-position: top center` to keep face in frame |
| Player panel (left side slide-in) conflicts with new grid | Panel still works — it pushes the grid right when open; grid reflows |
| 4-col grid too cramped on laptop | Responsive breakpoints at 1100/780/500px |
| `props-tabs` CSS specificity clash with global `.tab.active` | Use `.props-tabs .tab.active` (nested selector wins) |
| Old `.prop-card` CSS affects new card | New class is `.prop-card-v2` — completely separate, no conflict |
| `stat_line_text` for combos (PRA, Pts+Ast) | Detect combo stat and format as e.g. "Pts+Ast: Over 32.5" |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (planned directly from codebase context)
- GEMINI_SESSION: N/A
