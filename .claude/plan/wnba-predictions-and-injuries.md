# WNBA Predictions + Injuries Fix

## Diagnosis (evidence from live diagnostic)

**A'ja Wilson next-game prediction:**
- L20 avg: PTS 27.4, AST 3.4, REB 10.2 (per current parquet)
- Raw model output tonight @ MIN: **PTS 23.5, AST 2.6, REB 8.5** (reasonable)
- Calibration offsets: PTS −3.472, AST −1.739, REB −0.311 (model *under*-predicts historically)
- Card formula `raw − offset` produces: **PTS 26.97, AST 4.34, REB 8.81** ✓ Sensible.
- User's screenshot shows 15.3/1.9/4.8. This can only happen if the running dashboard has stale models cached in memory from a boot BEFORE the P9 retrain. A dashboard restart fixes it — but we should also add defense-in-depth so a bad model output never surfaces again.

**Injuries:**
- `https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries` now returns **403 Forbidden** (was working earlier — ESPN added blocking).
- Cache is empty (`0 players across 0 teams`) so every player looks "Active".
- No injury-based filter exists in `generate_wnba_props` — even if we HAD injury data, we'd still show OUT players on the props page.
- Name-lookup is exact-match only — Skylar Diggins vs "Skylar Diggins-Smith" would miss even if the source works.

## Task Type
- [x] Fullstack — backend (predict/inj) + minor frontend polish

## Technical Solution

### Predictions

1. **Apply `_blend_projection` in the dashboard prediction card** — mirrors the props generator so a wild model output can never dominate. Blend weight α = 0.55 (slightly more anchored to L20 than props' 0.65, since this is a headline "next game" number the user sees directly).

2. **Show model↔form context in the card subtitle**: `Tonight: @ MIN · L20 27.4 PTS · form ↑ (L5 32.2)` so the user can see the projection reference points at a glance.

3. **Rebuild calibration from clean data**: The current offsets were computed off ~30 seed predictions from Aug 2 debug work. That's not real accumulation. Delete `data/wnba/model_calibration.json` and let it re-accumulate organically. Also raise the calibration threshold from 25 to 40 samples per stat so a small biased sample can't dominate.

4. **Prediction confidence badge** (HIGH/MED/LOW) driven by (a) sample count in L20 and (b) |projection − L20 avg| tightness.

### Injuries

1. **Robust ESPN fetch**: rotate User-Agent, add Referer header (ESPN often 403s bare requests). If still 403, fall back to HTML scraping of `https://www.espn.com/wnba/injuries` (parse the same page ESPN's public UI serves).

2. **Persist last-good injury cache to disk** (`data/wnba/injuries_cache.json`) — same pattern used for the odds API. When live fetch fails, use last-known good data (with a "cached N hours ago" indicator).

3. **Fuzzy player-name lookup**: try exact match → last-name+first-initial → last-name only. Handles "Skylar Diggins" vs "Skylar Diggins-Smith" variants.

4. **Filter OUT/DOUBTFUL players from `generate_wnba_props`**: new param `exclude_injured=True`, checks injury feed by player name, drops player entirely if status ∈ {OUT, DOUBTFUL, "Out for Season"}.

5. **Prominent injury card**: for OUT/DOUBTFUL players on the player analysis page, show red status card BEFORE the prediction card and gray out the prediction values (they're not going to play).

## Implementation Steps

1. `utils/wnba_injuries.py` — HTML fallback, disk cache, fuzzy lookup, better UA.
2. `utils/wnba_props.py` — new `exclude_injured` param; filter injured players.
3. `dashboard/app.py::_wnba_prediction_card` — apply `_blend_projection`, add form subtitle, apply confidence pill.
4. `dashboard/app.py::_wnba_injury_card` — prominent OUT card + tighter status color.
5. `utils/wnba_prediction_tracker.py` — raise `min_samples` default 25 → 40.
6. **Delete stale calibration**: `rm data/wnba/model_calibration.json` (regenerates automatically once enough graded predictions exist).
7. **Tests**: fuzzy lookup, exclude_injured filter, blending in card.
8. Verify: boot dashboard, hit /wnba/, check A'ja shows ~26.9 PTS + form subtitle, injuries populate.
9. Commit + push to GitHub + HF Space.

## Key Files

| File | Op | Purpose |
|---|---|---|
| `utils/wnba_injuries.py` | Modify | Fallback HTML scrape + disk cache + fuzzy name lookup + retry |
| `utils/wnba_props.py` | Modify | `exclude_injured` param, integrate injury check |
| `dashboard/app.py` | Modify | Blend in prediction card, form subtitle, OUT-status injury card |
| `utils/wnba_prediction_tracker.py` | Modify | Raise calibration min_samples |
| `data/wnba/model_calibration.json` | Delete | Bad seed-data offsets |
| `data/wnba/injuries_cache.json` | Create | Disk fallback for injury feed |
| `tests/test_wnba_injuries.py` | Modify | Add fuzzy lookup + HTML fallback tests |
| `tests/test_wnba_props.py` | Modify | Add exclude_injured filter tests |

## Risks and Mitigation

| Risk | Mitigation |
|---|---|
| ESPN also blocks HTML scraping | Cache last-good indefinitely; provide manual override JSON file the user can populate |
| Fuzzy name match creates false positives | Only allow fuzzy match on unique last names; require min 2 chars first-name match |
| Blending makes predictions too conservative | α=0.55 not 0.35; still preserves 55% of model signal |
| Deleting calibration file resets accuracy tracking | We keep prediction_history.json — only calibration offsets reset |
| exclude_injured drops too many players | Only OUT/DOUBTFUL/Season-ending; QUESTIONABLE still shows with a flag |

## Estimated: 60 min work. 5 files touched, ~8 new tests.

## SESSION_ID
- CODEX_SESSION: N/A (local wrapper not installed)
- GEMINI_SESSION: N/A
