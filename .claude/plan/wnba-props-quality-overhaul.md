# WNBA Props Quality Overhaul

## Ultrathink Root-Cause Analysis

**Symptom (user):** Props page shows all UNDERs. Courtney Williams REB projected 1.0 when she averages 4.5. Bench players cluttering the list. No REB+AST filter.

**Empirical findings (from live diagnostic on 65 high-usage WNBA players):**

| Aspect | Finding |
|---|---|
| Model bias vs L20 avg | 0/25 top-minute players had PTS or REB predicted < 70% of their L20 average |
| Real OVERs from the model | Present — Stewart 22.6→25.8, Plum 21.1→25.0, Mitchell 25.4→34.1, Ionescu 16.2→20.9, Sykes 17.6→23.6 |
| Courtney Williams REB | L20 = 5.2, model predicts 4.2 (-18%). *Not* 1.0 — the user was likely looking at cached data from before my last retrain, OR misread a combo prop |
| Where the "1.0 REB" prediction could come from | An older cache OR the synthetic-line for a low-minute row where all rolling averages are 0/NaN |

**Real root causes for the "all UNDERs" symptom:**

1. **Synthetic line has a systematic UPWARD bias.** Current logic: `round(avg * 2) / 2` then nudge integer-rounded lines up to `.5`. Player averaging 5.2 → line becomes 5.5 (nudged UP because avg > 5.0). This means the line sits *above* what the player typically does, so:
   - Historical UNDER hit rate ≈ 55–60%
   - Historical OVER hit rate ≈ 40–45%
   - UNDER picks win the EV race by construction, drowning out real OVERs
2. **No minutes-based player filter** — bench players show up because SYNTHETIC_MIN_LINE only checks the *stat* threshold, not overall playing time.
3. **REB+AST combo missing** from the odds fetcher, props generator, dashboard pills, and stat helpers.
4. **Model conservative-bias tail** — for a handful of top-usage stars (A'ja Wilson, Kayla McBride, Marina Mabrey) the model does project 15–28% below L20 avg. This is a mild regression-to-mean effect from training on all games including low-minute rows. Sample weighting fixes it.

## Task Type
- [x] Fullstack — projection blending + line construction + player filter + UI stat pill + optional retrain

## Technical Solution

**Layered fixes, ordered by impact:**

### 1. Fix synthetic-line bias (biggest single-lever fix)
Change the line-construction from "nudge up on integers" to a **median-anchored** approach:
```python
# Use L20 median, not mean → naturally 50/50 historical split
line_center = float(actuals.median())
# Round to nearest 0.5
line = round(line_center * 2) / 2
# For integer results, jitter based on the L5-vs-L20 trend (recent form)
# Rising trend (L5 > L20) → nudge DOWN (encourage OVER)
# Falling trend → nudge UP (encourage UNDER)
if line == int(line):
    l5_avg = _actual_stat_sum(recent.head(5), stat).mean()
    line = line - 0.5 if l5_avg > line else line + 0.5
```
This produces synthetic lines where OVERs and UNDERs compete on model+historical merits, not construction bias.

### 2. Blend model prediction with recent form (safety net for outliers)
Prevents catastrophic under-projections like "1.0 REB for a 4.5 avg player" even if they somehow slip through:
```python
final_projection = clip(
    0.65 * model_pred + 0.35 * L20_avg,   # trust model but anchor to reality
    lower = 0.4 * L20_avg,                # never below 40% of recent form
    upper = 1.6 * L20_avg,                # never above 160% either
)
```
This is defense-in-depth: even if the model temporarily produces a bad prediction (feature-vector edge case, out-of-distribution matchup), the projection stays sane.

### 3. Minutes-based player filter (≥ 15 MPG)
Only include a player in the synthetic props pool when their L20 avg minutes ≥ 15. Bench players are dropped entirely. This is *in addition to* the SYNTHETIC_MIN_LINE stat floors.

### 4. Add REB+AST combo everywhere
- `wnba_props._COMBO_COMPONENTS["REB+AST"] = ["REB", "AST"]`
- `_SYNTHETIC_MIN_LINE["REB+AST"] = 5.5`
- `wnba_odds_fetcher.MARKET_TO_STAT["player_rebounds_assists"] = "REB+AST"`
- Dashboard props page stat filter pills: add `REB+AST`
- Dashboard player analysis: add `R+A` to `_WNBA_STAT_COMPONENTS` (already has P+R, P+A, PRA)

### 5. Retrain WNBA models with minutes-weighted samples
Fixes the mild bias for outlier stars (A'ja, McBride). In `train_improved_models.py`:
```python
sample_weight = training_df["MIN"].clip(1, 40) / 40.0
model.fit(X, y, sample_weight=sample_weight)
```
Combined with filtering training rows to `MIN >= 8` (removes garbage-time noise).

## Implementation Steps

1. **`utils/wnba_props.py`**
   - Change `_synthetic_line_from_recent` to use median + trend-aware jitter
   - Add `_blend_projection(model_pred, l20_avg)` helper with clip bounds
   - In `_build_prop`, apply blending: `projected = _blend_projection(projected, l20_avg)`
   - Add `min_avg_min: float = 15.0` param to `generate_wnba_props`; skip players whose L20 avg minutes < threshold
   - Add `"REB+AST": ["REB", "AST"]` to `_COMBO_COMPONENTS` and `5.5` to `_SYNTHETIC_MIN_LINE`
   - Include `"REB+AST"` in the `stats_available` loop

2. **`utils/wnba_odds_fetcher.py`**
   - Add `"player_rebounds_assists": "REB+AST"` to `MARKET_TO_STAT`
   - `_ALL_MARKETS` picks it up automatically

3. **`dashboard/app.py`**
   - Props filter pills: add `REB+AST` after `PTS+AST`
   - Player-analysis stat pills: add `R+A` to `_WNBA_STATS` and `_WNBA_STAT_COMPONENTS`
   - Color palette: assign one for `REB+AST` and `R+A`

4. **`scripts/train_improved_models.py`**
   - Filter training rows: `df = df[df["MIN"] >= 8]`
   - Pass `sample_weight = df["MIN"].clip(1, 40) / 40.0` to `.fit()`
   - Retrain: `python scripts/train_improved_models.py --league wnba`

5. **Tests** (`tests/test_wnba_props.py`)
   - Median-based synthetic line rounds correctly and jitters on trend
   - Blending clamps to [0.4×avg, 1.6×avg]
   - MIN ≥ 15 filter drops bench players
   - REB+AST combo generates props correctly

6. **Verify + push**
   - Boot dashboard, hit `/wnba/props`, confirm OVER/UNDER split is ~40/60 or better (not 100/0)
   - Confirm bench players gone, REB+AST filter pill works
   - Full pytest suite green
   - Commit + push to GitHub and Hugging Face (LFS-safe patterns already in place)

## Key Files

| File | Op | Purpose |
|---|---|---|
| `utils/wnba_props.py` | Modify | Median-based line, projection blending, MIN filter, REB+AST combo |
| `utils/wnba_odds_fetcher.py` | Modify | Add player_rebounds_assists market |
| `dashboard/app.py` | Modify | REB+AST pill on props page + R+A on player analysis |
| `scripts/train_improved_models.py` | Modify | Sample weights + MIN>=8 filter |
| `models/wnba/*.pkl` | Regenerate | Retrained with weighting |
| `tests/test_wnba_props.py` | Modify | Add 4 tests (median line, blending clamp, MIN filter, REB+AST) |

## Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Median-jitter still favors UNDER for cold streaks | The trend check (L5 vs L20) reverses jitter direction on rising form; over time the split converges to ~50/50 |
| Blending destroys good model signal | 65% weight on model preserves matchup intelligence; clamp only fires for extreme outliers |
| MIN ≥ 15 filter is too aggressive for early-season bench players emerging as starters | Threshold applied to L20 avg only — a player who starts trending upward crosses 15 MPG naturally once they're used |
| Retrain with sample weights makes bench predictions worse | Acceptable — we don't surface bench predictions on the props page anyway |
| REB+AST market may not be available from The Odds API for WNBA | If empty, synthetic-line path picks it up automatically (already generalized) |

## Ordering & Time Estimate
- Step 1 (props helper): 30 min
- Step 2 (odds market): 5 min
- Step 3 (dashboard UI): 15 min
- Step 4 (retrain): 5 min (mostly waiting on training)
- Step 5 (tests): 20 min
- Step 6 (verify + push): 15 min

**Total: ~90 min of focused work.**

## SESSION_ID
- CODEX_SESSION: N/A (multi-model wrapper not installed locally; plan authored by Claude with empirical diagnostic evidence)
- GEMINI_SESSION: N/A
