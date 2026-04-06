# Implementation Plan: Player Analysis Page — Final Design (from zip)

## Source Reference
`/Users/rodgersbahati/Downloads/zip.zip` → `src/App.tsx`

---

## Design Token Extraction

### Colors
| Token | Value |
|-------|-------|
| Body bg | `#0B101A` |
| Card bg | `#131A2A` |
| Card border | `rgba(30,41,59,0.8)` (`slate-800/80`) |
| Text primary | `#f1f5f9` (slate-200) |
| Text secondary | `#94a3b8` (slate-400) |
| Text muted | `#64748b` (slate-500) |
| Teal accent | `#2DD4BF` |
| Rose/coral | `#FB7185` |
| Amber | `#F59E0B` |
| Grid lines | `#1E293B` |

---

## Step-by-Step Implementation

### Step 1 — CSS: Core design tokens
**File**: `dashboard/assets/custom.css`

Update root variables + body:
```css
:root {
  --bg-root: #0B101A;
  --bg-card: #131A2A;
  --border-card: rgba(30,41,59,0.8);
  --teal: #2DD4BF;
  --rose: #FB7185;
  --amber: #F59E0B;
}
html, body { background: #0B101A; }
```

Update `.analysis-card`:
```css
.analysis-card {
  background: #131A2A;
  border: 1px solid rgba(30,41,59,0.8);
  border-radius: 16px;   /* rounded-2xl */
  padding: 24px;
  margin-bottom: 0;      /* gaps handled by parent flex gap */
  box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
```

---

### Step 2 — CSS: Navbar
**File**: `dashboard/assets/custom.css`

```css
.navbar {
  background: rgba(11,16,26,0.8);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(30,41,59,0.5);
  position: sticky;
  top: 0;
  z-index: 50;
}
.navbar-brand {
  background: linear-gradient(to right, #2DD4BF, #3B82F6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -0.02em;
}
/* Active nav pill */
.nav-link.active {
  background: rgba(45,212,191,0.10);
  color: #2DD4BF;
  border: 1px solid rgba(45,212,191,0.20);
  border-radius: 9999px;
  padding: 8px 20px;
  box-shadow: 0 0 15px rgba(45,212,191,0.10);
}
/* Inactive nav */
.nav-link {
  color: #94a3b8;
  padding: 8px 20px;
  border-radius: 9999px;
}
.nav-link:hover { color: #f1f5f9; }
```

---

### Step 3 — CSS + Layout: Player header
**File**: `dashboard/assets/custom.css` + `app.py` `update_player_header`

**Photo glow effect** (CSS):
```css
.player-photo-wrapper {
  position: relative;
  display: inline-block;
  flex-shrink: 0;
}
.player-photo-wrapper::before {
  content: "";
  position: absolute;
  inset: -4px;
  background: linear-gradient(135deg, #2DD4BF, #3B82F6);
  border-radius: 50%;
  filter: blur(8px);
  opacity: 0.4;
  z-index: 0;
}
.player-photo-circle {
  position: relative;
  z-index: 1;
  width: 64px; height: 64px;
  border-radius: 50%;
  border: 2px solid rgba(51,65,85,0.8);
  object-fit: cover;
  object-position: top;
}
```

**Team badge** (CSS):
```css
.team-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 8px;
  border-radius: 6px;
  background: rgba(30,41,59,0.8);
  border: 1px solid rgba(51,65,85,0.5);
  color: #94a3b8;
  font-size: 0.72rem;
  font-weight: 600;
}
```

**GTD/Injury badge** — SOLID amber fill (not transparent):
```css
.injury-badge-gtd {
  background: #FBBF24;       /* solid amber-400 */
  color: #451a03;            /* amber-900 dark text */
  border: none;
  border-radius: 9999px;
  padding: 4px 12px;
  font-size: 0.72rem;
  font-weight: 700;
  box-shadow: 0 0 20px rgba(251,191,36,0.4);
}
```

**Stats row** (PTS/AST/REB inline):
```css
.player-stat-chips { display: flex; gap: 20px; margin-top: 8px; align-items: center; }
.player-stat-chip {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 0.85rem;
  background: transparent;
  border: none;
  padding: 0;
  color: #94a3b8;
}
.player-stat-chip span:first-child { color: #f1f5f9; font-weight: 600; }
.player-stat-chip.chip-ast span:first-child { color: #FB7185; }  /* AST = rose */
.player-stat-chip .chip-label { color: #94a3b8; font-weight: 400; }
```

In `update_player_header` Python:
- Add wrapper div `.player-photo-wrapper` around the `<img>`
- Give AST chip className `"player-stat-chip chip-ast"`

---

### Step 4 — CSS: Search bar
**File**: `dashboard/assets/custom.css`

```css
.player-search-bar {
  background: #131A2A;
  border: 1px solid rgba(30,41,59,1);
  border-radius: 9999px;    /* rounded-full */
  padding: 0 16px;
  height: 44px;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.player-search-bar:focus-within {
  border-color: rgba(45,212,191,0.5);
  box-shadow: 0 0 0 1px rgba(45,212,191,0.5);
}
```

Also update Dropdown CSS to remove its own border/bg:
```css
.player-search-dropdown .Select-control {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
```

---

### Step 5 — CSS: Stat type filter tabs
**File**: `dashboard/assets/custom.css`

```css
.analysis-filters .tab {
  padding: 8px 20px;
  border-radius: 9999px;
  border: 1px solid transparent;
  font-size: 0.75rem;
  font-weight: 500;
  color: #94a3b8;
  background: transparent;
  transition: all 0.15s;
}
.analysis-filters .tab:hover {
  color: #f1f5f9;
  background: rgba(30,41,59,0.5);
}
.analysis-filters .tab.active {
  background: rgba(45,212,191,0.10);
  color: #2DD4BF;
  border-color: rgba(45,212,191,0.30);
  box-shadow: 0 0 10px rgba(45,212,191,0.10);
}
/* Time period tabs (second row) */
.analysis-filters .period-tab.active {
  background: rgba(30,41,59,1);
  color: #f1f5f9;
  border-color: rgba(51,65,85,0.5);
  box-shadow: none;
}
/* Separator line below period tabs */
.analysis-filters { border-bottom: 1px solid rgba(30,41,59,0.5); padding-bottom: 20px; margin-bottom: 24px; }
```

The period tabs need a second className `"tab period-tab"` in `create_player_analysis_page()`.

---

### Step 6 — CSS: Bar chart card
**File**: `dashboard/assets/custom.css`

The chart is inside `.analysis-card` — no extra class needed. But bar colors must be:
- Over threshold → `#2DD4BF` (teal)
- Under threshold → `#FB7185` (rose, not the current salmon)

In `update_main_chart` callback, change colors:
```python
COLORS["hit_yes"] = "#2DD4BF"
COLORS["hit_no"]  = "#FB7185"
```
Or update inline in the bar trace cell fill:
```python
color = "#2DD4BF" if val > threshold else "#FB7185"
```

Chart background: `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"`, grid `#1E293B`.

---

### Step 7 — CSS: Trend Insight box
**File**: `dashboard/app.py` inline styles (around line 789)

Change the trend insight box to:
```python
style={
  "background": "linear-gradient(to right, rgba(45,212,191,0.10), transparent)",
  "border": "1px solid rgba(45,212,191,0.20)",
  "borderLeft": "none",
  "borderRadius": "12px",
  "padding": "20px",
  "marginBottom": "20px",
  "position": "relative",
  "overflow": "hidden",
}
```
And the left glowing bar:
```python
html.Div(style={
  "position": "absolute", "left": "0", "top": "0", "bottom": "0",
  "width": "4px", "background": "#2DD4BF",
  "boxShadow": "0 0 10px rgba(45,212,191,0.5)", "borderRadius": "12px 0 0 12px",
}),
```

---

### Step 8 — CSS: Supporting stat mini-cards (5-col grid)
**File**: `dashboard/assets/custom.css`

```css
.supporting-stat-card {
  background: #131A2A;
  border: 1px solid rgba(30,41,59,0.8);
  border-radius: 12px;
  padding: 16px;
  transition: border-color 0.15s;
}
.supporting-stat-card:hover { border-color: rgba(51,65,85,1); }
.supporting-stat-label { font-size: 0.625rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; margin-bottom: 6px; }
.supporting-stat-value { font-size: 1.25rem; font-weight: 600; color: #f1f5f9; }
.supporting-stat-sub { font-size: 0.625rem; color: #64748b; margin-top: 4px; }
```

The `supporting-stats-cards` grid in app.py needs `"display":"grid","gridTemplateColumns":"repeat(5,1fr)","gap":"16px"`.

---

### Step 9 — CSS: Season Trends card
Dot style in `update_season_trends_chart`:
```python
line=dict(color="#2DD4BF", width=2.5),
mode="lines+markers",
marker=dict(size=8, color="#131A2A", line=dict(color="#2DD4BF", width=2))
# Reb trace uses #F59E0B
```
Legend dots: teal with glow (can do via Plotly legend marker symbol).

---

### Step 10 — CSS + Python: Matchup Analysis card
Remove the polar radar chart (not in the reference design — just the stats table).

Update `create_matchup_content()`:
- Remove `_build_defense_radar()` call and sub-card
- Keep only: title, defense header, sub-line, underline tabs, stats table
- Update stat row label color to `#cbd5e1` (slate-300)
- Stats header row uses `text-[10px] uppercase tracking-widest` style
- Tab active: `border-b-2 border-white text-white`, inactive: `text-slate-500`

---

### Step 11 — CSS + Python: Injury Context card
Update `create_injury_context_card()` return:
```python
html.Div([
  # Gradient overlay divs (absolute positioned)
  html.Div(style={
    "position":"absolute","inset":"0",
    "background":"linear-gradient(135deg, rgba(244,63,94,0.05), transparent, rgba(45,212,191,0.05))",
    "pointerEvents":"none",
  }),
  html.Div(style={
    "position":"absolute","right":"0","top":"0","bottom":"0","width":"8rem",
    "background":"linear-gradient(to left, rgba(244,63,94,0.05), transparent)",
    "pointerEvents":"none",
  }),
  # Content
  html.Div([
    html.H3("Injury Context", ...),
    html.P([html.Strong("Status:", style={"color":"#FB7185"}), " Questionable (Back Spasms)."]),
    html.P("Missed practice yesterday...", style={"background":"rgba(30,41,59,0.3)","padding":"12px","borderRadius":"8px","border":"1px solid rgba(30,41,59,0.5)"}),
    html.P([html.Strong("Impact:", style={"color":"#2DD4BF"}), " If active, minutes may be limited..."]),
  ], style={"position":"relative","zIndex":"1"}),
], style={"position":"relative","overflow":"hidden"}, className="analysis-card")
```

---

### Step 12 — CSS + Python: Prop Bet History card
Update `create_prop_bet_history_card()` rows:
- Hit prop pill: `bg-teal-500/10 border-teal-500/20 text-teal-400` → inline: `background:rgba(45,212,191,0.10), border:1px solid rgba(45,212,191,0.20), color:#2DD4BF`
- Miss prop pill: `bg-rose-500/10 border-rose-500/20 text-rose-400` → `rgba(251,113,133,0.10), rgba(251,113,133,0.20), #FB7185`
- Row hover: `background:rgba(30,41,59,0.20), borderRadius:8px`
- "Recent Game" button: `background:#1E293B, border:1px solid rgba(51,65,85,0.5), color:#cbd5e1`

---

## Key Files

| File | Operation | Lines |
|------|-----------|-------|
| `dashboard/assets/custom.css` | Modify | Root vars, card, nav, header, tabs, trend box, stat mini-cards |
| `dashboard/app.py` navbar section | Modify | Logo gradient, nav pill classes |
| `dashboard/app.py:update_player_header` | Modify | Photo wrapper, GTD solid fill, AST rose chip |
| `dashboard/app.py:create_player_analysis_page` | Modify | Period tab class `period-tab`, search bar rounded-full |
| `dashboard/app.py:update_main_chart` | Modify | Bar colors `#2DD4BF` / `#FB7185`, chart bg |
| `dashboard/app.py:~789 trend insight` | Modify | Inline styles for gradient + left glow bar |
| `dashboard/app.py:create_matchup_content` | Modify | Remove radar chart, fix stat row colors |
| `dashboard/app.py:create_injury_context_card` | Modify | Gradient overlays, status/impact colors |
| `dashboard/app.py:create_prop_bet_history_card` | Modify | Hit/miss pill styles, hover bg |
| `dashboard/app.py:update_season_trends_chart` | Modify | Dots style, amber for REB line |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
