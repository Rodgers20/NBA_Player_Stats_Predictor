# Plan: Props Prediction Report + Fix Non-ML Model Performance

## Background

### Issue 1 — Props Report Not Showing

The tracking infrastructure exists and is working:
- `utils/prediction_tracker.py` — `save_daily_props()`, `grade_props()`, `get_props_record()`
- `data/props_history.json` — stores daily snapshots + grades
- `data/prediction_report.xlsx` — exported Excel with "Props Record" sheet
- `/download-report` Flask endpoint — downloads the Excel

**But**: No dashboard UI renders this data. The three pages (Player Analysis, Today's Games,
Best Props) have no props record section. The user has no way to see performance inside the app.

### Issue 2 — Non-ML Props Performing Poorly

Two separate computation pipelines exist:

| Pipeline | Where used | Hit rate source | ML involved |
|----------|-----------|-----------------|-------------|
| `_compute_main_page_props` | Best Props main page | Raw L10 frequency | ❌ No |
| `_compute_sidebar_props` | Sidebar | L10 + ML prediction | ✅ Yes |

The main page pipeline uses only:
- `hit_rate = count(last_10 >= line) / 10` — raw frequency
- Lines based on median (good) but no contextual adjustment
- No blowout risk adjustment to the line itself (only to EV)
- No defense vs position weighting on the line

The sidebar ML pipeline:
- Calls `stat_predictor.predict_player_game()` → gets a contextual prediction
- Blends that into `calculate_smart_prop_score()`
- Result: more accurate predictions

**Fix**: Pass `get_predictor_fn` into `_compute_main_page_props` and use ML output to:
1. Adjust the line (blend ML prediction with median — like the sidebar does)
2. Compute a "contextual hit rate" that weighs recent form + ML delta

---

## Implementation Plan: Props Record Tab

### Task Type
- [x] Fullstack (Backend data → Frontend display)

### Solution

Add a third tab to the props page view switcher:

```
[Props]  [100% Alt Lines]  [Record ✓]
```

The "Record" view displays data from `get_props_record()` and `props_history.json`.

### Implementation Steps

#### Step 1 — Add "Record" tab to view switcher (`dashboard/app.py`)

In `create_best_props_page()`, add a third `html.Div` to `.props-view-switcher`:
```python
html.Div("Record ✓", id="props-view-tab-record", n_clicks=0, className="view-tab"),
```

Add `dcc.Store(id="props-view", data="props")` already exists — stays at 3 possible
values: `"props"`, `"alt"`, `"record"`.

#### Step 2 — Update view switcher callback (`dashboard/app.py`)

`update_props_view` currently handles 2 tabs. Extend to 3:
```python
@callback(
    [Output("props-view", "data"),
     Output("props-view-tab-props",   "className"),
     Output("props-view-tab-alt",     "className"),
     Output("props-view-tab-record",  "className"),
     Output("props-filter-panel",     "style")],
    [Input("props-view-tab-props",   "n_clicks"),
     Input("props-view-tab-alt",     "n_clicks"),
     Input("props-view-tab-record",  "n_clicks")],
    prevent_initial_call=True,
)
def update_props_view(n_props, n_alt, n_record):
    triggered = ctx.triggered_id
    if triggered == "props-view-tab-alt":
        return "alt",    "view-tab", "view-tab active", "view-tab", {"display": "none"}
    if triggered == "props-view-tab-record":
        return "record", "view-tab", "view-tab",        "view-tab active", {"display": "none"}
    return "props", "view-tab active", "view-tab", "view-tab", {}
```

#### Step 3 — Build `_create_props_record_section()` (`dashboard/app.py`)

New helper function that reads the live props record and renders it:

```python
def _create_props_record_section() -> html.Div:
    from utils.prediction_tracker import get_props_record
    record = get_props_record()
    # record = {
    #   "overall": {"graded": 120, "hits": 72, "hit_rate": 0.60},
    #   "by_stat": {"PTS": {...}, "AST": {...}, ...},
    #   "last_7_days": [...],
    #   "streak": {"current": 3, "type": "W"}
    # }

    overall = record.get("overall", {})
    graded = overall.get("graded", 0)
    hits   = overall.get("hits",   0)
    rate   = overall.get("hit_rate", 0.0)

    # 1. Summary card: Overall record
    # 2. By-stat breakdown table: PTS / AST / REB / 3PM / Combos
    # 3. Last 7 days performance strip
    # 4. Download Excel button (points to /download-report)
```

**Summary card layout:**
```
┌────────────────────────────────────────────────────────┐
│  PROPS RECORD                                          │
│  72 / 120  •  60.0% hit rate  •  W3 streak            │
└────────────────────────────────────────────────────────┘

┌──────┬──────────┬──────┬──────┬──────────┐
│ STAT │  GRADED  │  W   │  L   │ HIT RATE │
├──────┼──────────┼──────┼──────┼──────────┤
│ PTS  │    45    │  28  │  17  │   62%    │
│ AST  │    20    │  11  │   9  │   55%    │
│ REB  │    18    │  12  │   6  │   67%    │
│ 3PM  │    12    │   8  │   4  │   67%    │
│Combo │    25    │  13  │  12  │   52%    │
└──────┴──────────┴──────┴──────┴──────────┘

[Last 7 days strip: date | N props | hit rate]

[📥 Download Full Report]
```

**If no data yet** (props_history.json is empty `{}`):
- Show a message: "No props have been graded yet. Predictions are graded automatically after each game day."

#### Step 4 — Add `view == "record"` branch in `update_props_list` (`dashboard/app.py`)

```python
if view == "record":
    record_content = _create_props_record_section()
    return [record_content, _empty_counts, _default_lock_label]
```

#### Step 5 — CSS for record section (`dashboard/assets/custom.css`)

- `.props-record-summary` — overall stats banner (teal accent)
- `.props-record-table` — breakdown table (dark bg, row hover)
- `.props-record-streak` — W/L streak badge
- `.props-record-last7` — 7-day strip

---

## Implementation Plan: Fix Non-ML Prop Performance

### Solution

Pass `get_predictor_fn` into `_compute_main_page_props` and blend ML prediction into
the line and hit rate calculation. This mirrors what `_compute_sidebar_props` does but
without the full `calculate_smart_prop_score` machinery.

**Blend formula:**
```
ml_prediction = stat_predictor.predict_player_game(player_name, DF)
                .get(f"predicted_{stat.lower()}", None)

if ml_prediction is not None:
    blended_line = 0.6 * ml_prediction + 0.4 * median_stat
    # Use blended_line instead of pure median-based line
    # Recompute hit rate against the blended line
else:
    blended_line = line  # fallback to existing logic
```

This gives ML a 60% weight on the final line. The hit rate is then recomputed on the
blended line — props where ML disagrees with the recent average get a more conservative
line (harder to hit), filtering out over-confident picks.

### Implementation Steps

#### Step 6 — Add `get_predictor_fn` parameter to `_compute_main_page_props` (`utils/props_cache.py`)

Current signature:
```python
def _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info,
                              availability_map, players_to_analyze, game_spreads=None):
```

New signature:
```python
def _compute_main_page_props(DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info,
                              availability_map, players_to_analyze,
                              game_spreads=None, get_predictor_fn=None):
```

#### Step 7 — Use ML prediction for line blending (inside `_compute_main_page_props`)

After the median-based `line` is computed for each stat, add:

```python
# ML line blending — improves line accuracy vs. pure median
ml_line = None
if get_predictor_fn:
    try:
        predictor = get_predictor_fn(stat_type)
        if predictor:
            result = predictor.predict_player_game(player_name, DF)
            pred_key = f"predicted_{stat_type.lower()}"
            ml_pred = result.get(pred_key)
            if ml_pred and ml_pred > 0:
                ml_line = round((0.6 * ml_pred + 0.4 * float(line) - 0.5) * 2) / 2 + 0.5
                # recompute hits on blended line
                hits_over  = (recent_stats >= ml_line).sum()
                hits_under = (recent_stats <  ml_line + 1.0).sum()
                hit_rate_over  = hits_over  / n
                hit_rate_under = hits_under / n
                line = ml_line
    except Exception:
        pass  # fall back to median line silently
```

The rounding formula keeps the line at a proper `.5` increment.

#### Step 8 — Pass `get_predictor_fn` from `refresh_props_cache` to main props

In `refresh_props_cache`, change the `main_data` call:
```python
main_data = _compute_main_page_props(
    DF, PLAYER_POSITIONS, DEFENSE_VS_POS, game_info,
    availability_map, players_to_analyze,
    game_spreads=game_spreads,
    get_predictor_fn=get_predictor_fn,  # ← add this
)
```

#### Step 9 — Add `ml_used` flag to prop output (optional, for record tracking)

In `_make_prop()`, add `"ml_used": ml_line is not None` to distinguish ML-enhanced
props in the tracking data. This lets the record table break down "ML-enhanced" vs
"median-only" props separately in future.

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/app.py` — view switcher | Modify | Add "Record ✓" third tab |
| `dashboard/app.py` — view callback | Modify | Handle 3 tabs, add props-view-tab-record output |
| `dashboard/app.py` — new helper | Add | `_create_props_record_section()` |
| `dashboard/app.py` — update_props_list | Modify | Add `view == "record"` branch |
| `dashboard/assets/custom.css` | Modify | Record section table/summary styles |
| `utils/props_cache.py` — signature | Modify | Add `get_predictor_fn=None` to `_compute_main_page_props` |
| `utils/props_cache.py` — line logic | Modify | ML blending inside `_compute_main_page_props` |
| `utils/props_cache.py` — refresh | Modify | Pass `get_predictor_fn` to `_compute_main_page_props` |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| ML models not loaded at startup → NoneType error | `get_predictor_fn` returns None for unloaded models; all ML calls wrapped in try/except |
| ML blending slows down props cache refresh | ML `predict_player_game` is fast (seconds total for ~300 players); acceptable overhead |
| Record section shows empty state on day 1 | Guard with `if graded == 0:` → friendly empty state message |
| `props_history.json` currently empty `{}` | Will populate after first game day; existing grade scheduler handles this |
| Blended line changes existing props significantly | 60/40 blend is conservative; line can move at most ±15% from median |
