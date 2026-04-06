# Implementation Plan: Props Page Navigation Overhaul

## Task Type
- [x] Frontend (UI/UX focused)

## Problem Statement
1. **No direction filter** — Overs and Unders are mixed together; no way to see only one type
2. **Long scroll to Alt Lines** — the 100% Alt Lines section sits below ALL prop cards, requiring a long scroll
3. **No section jumping** — once scrolled deep, getting back to filters is painful

## Technical Solution

Introduce a **two-level navigation system**:

### Level 1 — Primary View Switcher (new)
Big pill tabs at the top of the content area:
```
[ PROPS ]   [ 100% ALT LINES ]
```
- Backed by `dcc.Store(id="props-view", data="props")`
- Clicking "100% ALT LINES" replaces the props list with the alt lines table instantly (no scroll)
- Clicking "PROPS" returns to the main props list

### Level 2 — Direction Filter (added to existing filter row)
Inside the PROPS view, add an Over/Under toggle before the stat tabs:
```
[ ALL ] [ OVERS ↑ ] [ UNDERS ↓ ]
```
- Backed by `dcc.Store(id="props-direction-filter", data="all")`
- "OVERS" shows only `direction == "Over"` props
- "UNDERS" shows only `direction == "Under"` props (styled red/orange to visually separate)

---

## Implementation Steps

### Step 1 — Add `dcc.Store` + `dcc.Store` for view & direction
In `create_best_props_page()`, add alongside existing stores:
```python
dcc.Store(id="props-view", data="props"),
dcc.Store(id="props-direction-filter", data="all"),
```

### Step 2 — Add primary view switcher tabs HTML
Insert ABOVE the stat filter row:
```python
html.Div([
    html.Div("Props",            id="props-view-tab-props",    n_clicks=0, className="view-tab active"),
    html.Div("100% Alt Lines",   id="props-view-tab-alt",      n_clicks=0, className="view-tab"),
], className="props-view-switcher"),
```
Style: larger, bolder tabs with a teal underline indicator for active state.

### Step 3 — Add direction filter tabs HTML
Insert as a new row between the view switcher and stat tabs:
```python
html.Div([
    html.Div("All",     id="props-dir-all",   n_clicks=0, className="tab active"),
    html.Div("Overs ↑", id="props-dir-over",  n_clicks=0, className="tab tab-over"),
    html.Div("Unders ↓",id="props-dir-under", n_clicks=0, className="tab tab-under"),
    html.Div(style={"flex": "1"}),  # right-align remaining row-2 controls
    # ... existing game + sort dropdowns stay here
], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px", "flexWrap": "wrap"}),
```

### Step 4 — Callback: `update_props_view` (new)
Controls the view Store and tab active states:
```python
@callback(
    [Output("props-view", "data"),
     Output("props-view-tab-props", "className"),
     Output("props-view-tab-alt", "className")],
    [Input("props-view-tab-props", "n_clicks"),
     Input("props-view-tab-alt",   "n_clicks")],
    prevent_initial_call=True,
)
def update_props_view(n_props, n_alt):
    triggered = ctx.triggered_id
    if triggered == "props-view-tab-alt":
        return "alt", "view-tab", "view-tab active"
    return "props", "view-tab active", "view-tab"
```

### Step 5 — Callback: `update_direction_filter` (new)
```python
@callback(
    [Output("props-direction-filter", "data"),
     Output("props-dir-all",   "className"),
     Output("props-dir-over",  "className"),
     Output("props-dir-under", "className")],
    [Input("props-dir-all",   "n_clicks"),
     Input("props-dir-over",  "n_clicks"),
     Input("props-dir-under", "n_clicks")],
    prevent_initial_call=True,
)
def update_direction_filter(n_all, n_over, n_under):
    triggered = ctx.triggered_id
    if triggered == "props-dir-over":
        return "Over", "tab", "tab tab-over active", "tab tab-under"
    if triggered == "props-dir-under":
        return "Under", "tab", "tab tab-over", "tab tab-under active"
    return "all", "tab active", "tab tab-over", "tab tab-under"
```

### Step 6 — Extend `update_props_list` callback
Add new inputs to the existing `update_props_list` function signature:
```python
@callback(
    Output("props-list", "children"),
    [Input("props-location-filter", "data"),
     Input("props-game-filter", "value"),
     Input("props-sort-dropdown", "value"),
     Input("props-data-store", "data"),
     Input("props-stat-filter", "data"),
     Input("props-direction-filter", "data"),   # NEW
     Input("props-view", "data")],              # NEW
)
def update_props_list(location_filter, game_filter, sort_by, props_data,
                      stat_filter, direction_filter, view):
    # NEW: if view == "alt", render the alt lines table
    if view == "alt":
        from utils.props_cache import get_cached_props
        alt_lines = get_cached_props().get("alt_lines_data", [])
        return _create_alt_lines_section(alt_lines)

    # existing filter logic unchanged...

    # NEW: direction filter (after stat filter, before game filter)
    if direction_filter and direction_filter != "all":
        props_data = [p for p in props_data if p.get("direction") == direction_filter]

    # ... rest of existing logic unchanged
```

### Step 7 — CSS additions (`dashboard/assets/custom.css`)

```css
/* ── Primary view switcher ───────────────────────────────────── */
.props-view-switcher {
  display: flex;
  gap: 4px;
  margin-bottom: 20px;
  border-bottom: 2px solid rgba(255,255,255,0.06);
  padding-bottom: 0;
}
.view-tab {
  padding: 10px 22px;
  font-size: 0.95rem;
  font-weight: 700;
  color: #4a5a75;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  margin-bottom: -2px;
  transition: color 150ms, border-color 150ms;
  letter-spacing: 0.02em;
  user-select: none;
}
.view-tab:hover {
  color: #8ca0c0;
}
.view-tab.active {
  color: #2dd4bf;
  border-bottom-color: #2dd4bf;
}

/* ── Direction filter tabs ───────────────────────────────────── */
.tab.tab-over.active  { background: rgba(34,197,94,0.15)  !important; color: #22c55e !important; border-color: rgba(34,197,94,0.4) !important; }
.tab.tab-under.active { background: rgba(239,68,68,0.12)  !important; color: #ef4444 !important; border-color: rgba(239,68,68,0.4) !important; }
.tab.tab-over:hover   { color: #22c55e; }
.tab.tab-under:hover  { color: #ef4444; }
```

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/app.py` | Modify `create_best_props_page()` | Add 2 stores + view switcher HTML + direction filter row HTML |
| `dashboard/app.py` | Add `update_props_view` callback | View tab switching logic |
| `dashboard/app.py` | Add `update_direction_filter` callback | Direction filter logic |
| `dashboard/app.py` | Modify `update_props_list` callback | Add `direction_filter` + `view` inputs, render alt-lines when view="alt" |
| `dashboard/assets/custom.css` | Add rules | View tab styles + direction filter colors |

## Before/After UX

| Before | After |
|--------|-------|
| One long scrollable page | Two primary views: PROPS / 100% ALT LINES |
| Must scroll ~200px to reach Alt Lines | Click "100% Alt Lines" tab → instant render |
| Overs + Unders mixed, no way to separate | Direction filter: ALL / OVERS ↑ / UNDERS ↓ |
| Filter bar disappears on scroll | View switcher prominent, always at top of content |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `update_props_list` now has 7 inputs instead of 5 — callback signature mismatch | Carefully update both `@callback` decorator and function signature together |
| Alt lines not loaded until cache warms | Re-read cache on alt-view render (same pattern as interval refresh) |
| `prevent_initial_call` on new callbacks — initial view state correct | Both stores initialize with `data="props"` / `data="all"` so no stale state |

## SESSION_ID
- CODEX_SESSION: N/A (no external model calls)
- GEMINI_SESSION: N/A (no external model calls)
