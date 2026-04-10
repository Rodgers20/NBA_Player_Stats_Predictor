# Implementation Plan: Props Quality + EV Overhaul

## Problem Statement

Three related bugs:

1. **Line is set from L10 median** — if a player had 5 injury/rest games in positions 6-10 of their last 10, the median is dragged way down. Barrett averaging 20.5 PPG over L5 shows as "Over 10.5" because the L10 median is ~10. The line should reflect current form.

2. **EV is identical to Hit Rate** — `ev = calculate_ev(hit_rate)` at constant -110 odds is a linear function of hit_rate. They rank props identically. True +EV should be: *the gap between our model's probability estimate and the sportsbook's implied probability*. A line set far below a player's expected value (Barrett at 16.5 when he avg 20.5) is genuinely +EV because P(hit) >> implied probability.

3. **Too many picks, no quality gate** — every player who crosses 52% hit_rate at the model-generated line gets a prop. Need strict quality control.

---

## Technical Solution

### Core Insight

The existing flow has a layered problem:

```
Step A: Line = math.floor(L10_median) + 0.5         ← WRONG: L10 median distorted by old games
Step B: hit_rate = (L10 games >= line).sum() / 10    ← measured against wrong line
Step C: ev = calculate_ev(hit_rate)                  ← identical to hit_rate at -110
Step D: live odds overwrite line                     ← DOESN'T recalculate hit_rate!
Step E: ev recalculated using hit_rate at old line   ← hit_rate is now for wrong line
```

The correct flow should be:

```
Step A: baseline_line = math.floor(L5_avg) + 0.5     ← current form signal
Step B: ML blend on L5-based line (keep existing)
Step C: hit_rate = (L10 games >= model_line).sum()   ← historical frequency at model line
Step D: live odds overwrite prop["line"]             ← actual sportsbook line
Step E: RECALCULATE hit_rate at sportsbook line      ← hit historical at REAL line
Step F: model_prob = P(score > sb_line | pred=L5_avg) ← probability from model perspective
Step G: ev = calculate_ev(model_prob, sb_price)      ← real edge vs sportsbook
```

---

## Implementation Steps

### Step 1 — Use L5 Average as Primary Line Baseline
**File:** `utils/props_cache.py`, lines ~285-315

**Current:**
```python
avg_stat    = recent_stats.mean()
median_stat = recent_stats.median()

if stat_type == "PTS":
    line = math.floor(median_stat) + 0.5 if median_stat > 5 else 4.5
```

**Change to:**
```python
avg_stat    = recent_stats.mean()
median_stat = recent_stats.median()

# L5 form average — primary signal for current performance level
l5_avg = float(recent_stats.head(5).mean()) if len(recent_stats) >= 5 else float(avg_stat)

# Use L5 avg as the baseline (reflects current form, not historical median)
if stat_type == "PTS":
    line = math.floor(l5_avg) + 0.5 if l5_avg > 5 else 4.5
elif stat_type == "FG3M":
    line = math.floor(l5_avg) + 0.5 if l5_avg > 1 else 0.5
else:  # AST, REB
    line = math.floor(l5_avg) + 0.5 if l5_avg > 2 else 1.5
```

**Why:** L5 average captures current form. If Barrett scores 10, 10, 20, 22, 20 in L10 (old games pulled him down), L5 gives ~20.5. The line should be ~20.5, not 10.5.

**Expected deliverable:** Line for active/hot players now reflects their current scoring level.

---

### Step 2 — Store L5 Avg + Model Prediction in Prop Dict
**File:** `utils/props_cache.py`, lines ~301-315 (ML blending block) and ~391-410 (return dict)

**In the ML blending block**, capture the ml_pred value for later use:
```python
ml_pred_stored = None  # capture for EV calc

if get_predictor_fn:
    try:
        predictor = get_predictor_fn(stat_type)
        if predictor:
            ml_result = predictor.predict_player_game(player_name, DF)
            ml_pred = ml_result.get(f"predicted_{stat_type.lower()}")
            if ml_pred and float(ml_pred) > 0:
                ml_pred_stored = float(ml_pred)
                blended = 0.6 * float(ml_pred) + 0.4 * float(line)
                line = math.floor(blended) + 0.5
    except Exception:
        pass
```

**Add to prop dict** (in `_make_prop` return dict):
```python
"l5_avg":     round(l5_avg, 1),
"model_pred": round(ml_pred_stored, 1) if ml_pred_stored else round(l5_avg, 1),
```

`model_pred` defaults to `l5_avg` when no ML model is loaded — ensuring the EV calculation always has a meaningful prediction.

---

### Step 3 — Separate EV from Hit Rate in Live Odds Enrichment
**File:** `utils/props_cache.py`, lines ~498-523 (live odds block)

**Current (broken):**
```python
if s_odds:
    prop["line"] = s_odds["line"]          # overwrites line
    prop["ev"] = calculate_ev(prop["hit_rate"], over_american=s_odds["over_price"])
    # hit_rate is still from the OLD model-generated line!
```

**Replace with:**
```python
if s_odds:
    sb_line      = s_odds["line"]
    sb_over_price  = s_odds.get("over_price", -110)
    sb_under_price = s_odds.get("under_price", -110)

    prop["live_line"]        = sb_line
    prop["live_over_price"]  = sb_over_price
    prop["live_under_price"] = sb_under_price
    prop["live_bookmaker"]   = s_odds["bookmaker"]
    prop["has_live_odds"]    = True
    prop["line"]             = sb_line  # display the real sportsbook line

    # Recalculate hit rate at the ACTUAL sportsbook line
    player_key = prop["player"]
    stat_key   = prop["stat"]
    p_df = DF[DF["PLAYER_NAME"] == player_key].sort_values("_date", ascending=False)
    if "SEASON" in p_df.columns:
        cs = p_df[p_df["SEASON"].str.startswith("2025", na=False)]
        r10 = cs.head(10) if len(cs) >= 5 else p_df.head(10)
    else:
        r10 = p_df.head(10)

    if stat_key in r10.columns and not r10.empty:
        vals = pd.to_numeric(r10[stat_key], errors="coerce").dropna()
        if len(vals) >= 3:
            if prop["direction"] == "Over":
                sb_hits    = int((vals >= sb_line).sum())
                sb_hit_rate = sb_hits / len(vals)
            else:
                sb_hits    = int((vals < sb_line).sum())
                sb_hit_rate = sb_hits / len(vals)
            prop["hit_rate"] = round(sb_hit_rate, 4)
            prop["hits"]     = sb_hits
            prop["total"]    = len(vals)

    # EV = model's probability vs. sportsbook's implied probability
    # model_pred was stored in Step 2; use it to estimate true P(hit)
    model_pred_val = prop.get("model_pred") or prop.get("l5_avg") or prop.get("avg")
    if model_pred_val:
        std_val = float(DF[DF["PLAYER_NAME"] == player_key][stat_key].std(skipna=True) or 4.0)
        sb_price_for_dir = sb_over_price if prop["direction"] == "Over" else sb_under_price
        model_prob = calculate_hit_probability(
            prediction=float(model_pred_val),
            line=float(sb_line),
            std_dev=std_val,
            direction=prop["direction"].lower(),
            stat_type=stat_key,
        )
        prop["ev"]           = calculate_ev(model_prob, over_american=sb_price_for_dir)
        prop["model_prob"]   = round(model_prob, 4)

        # Implied probability from sportsbook price
        if sb_price_for_dir > 0:
            implied = sb_price_for_dir / (sb_price_for_dir + 100)
        else:
            implied = abs(sb_price_for_dir) / (abs(sb_price_for_dir) + 100)
        prop["implied_prob"] = round(implied, 4)
        prop["edge"]         = round(model_prob - implied, 4)  # raw probability edge
    else:
        prop["ev"] = calculate_ev(prop["hit_rate"], over_american=sb_price_for_dir)
        prop["model_prob"]   = prop["hit_rate"]
        prop["implied_prob"] = 0.524
        prop["edge"]         = prop["hit_rate"] - 0.524
else:
    # No live odds: EV from model_prob vs default -110 implied
    model_pred_val = prop.get("model_pred") or prop.get("l5_avg") or prop.get("avg")
    std_val = 4.0  # fallback
    if model_pred_val:
        model_prob = calculate_hit_probability(
            prediction=float(model_pred_val),
            line=float(prop["line"]),
            std_dev=std_val,
            direction=prop["direction"].lower(),
            stat_type=prop["stat"],
        )
        prop["ev"]           = calculate_ev(model_prob)
        prop["model_prob"]   = round(model_prob, 4)
        prop["implied_prob"] = 0.524
        prop["edge"]         = round(model_prob - 0.524, 4)
    prop["has_live_odds"]    = False
    prop["live_line"]        = None
    prop["live_over_price"]  = None
    prop["live_under_price"] = None
    prop["live_bookmaker"]   = None
```

**NOTE on performance:** The `DF` lookup per-prop is slightly expensive for 100+ props. Precompute a player→recent_stats dict before the enrichment loop to avoid scanning the entire DF per prop.

---

### Step 4 — Quality Gate: Reduce Quantity
**File:** `utils/props_cache.py`, lines ~525-536 (after live odds enrichment, before return)

**Add after the enrichment loop:**
```python
# ── Quality gate ──────────────────────────────────────────────────────
_MIN_EV        = 0.02     # 2% edge minimum (model_prob must exceed implied by ≥2pp)
_MIN_MODEL_PROB = 0.58    # model must give ≥58% probability
_MAX_PER_PLAYER = 2       # max 2 props per player
_HARD_CAP       = 25      # max total props shown

# Filter by minimum model probability (true quality bar)
props_data = [
    p for p in props_data
    if p.get("model_prob", 0) >= _MIN_MODEL_PROB
    and p.get("ev", 0) >= _MIN_EV
]

# Deduplicate per player — keep top 2 props by EV
from collections import defaultdict
per_player: dict = defaultdict(list)
for p in props_data:
    per_player[p["player"]].append(p)

deduped = []
for player_props in per_player.values():
    player_props.sort(key=lambda x: -x.get("ev", 0))
    deduped.extend(player_props[:_MAX_PER_PLAYER])

# Re-sort: locks first, then EV descending
deduped.sort(key=lambda x: (not x.get("is_lock", False), -x.get("ev", 0)))
props_data = deduped[:_HARD_CAP]
```

---

### Step 5 — Display: Separate EV and Hit Rate on Prop Cards
**File:** `dashboard/app.py`, function `update_props_list()`, ~lines 2660-2760

**Current prop card "EV badge" shows hit_rate as if it were EV.**

Changes to prop card:

1. **Hit Rate section** — keep as-is, show `hit_rate` as percentage with hits/total

2. **EV / Edge section** — redesign to show:
   - When live odds available: `"+EV: {edge*100:.0f}%" ` (e.g., "+EV: 18%")
   - Model prob vs implied: `"Model: {model_prob*100:.0f}% vs {implied_prob*100:.0f}% implied"`
   - Color: green if edge > 0.08, yellow if 0.02-0.08, not shown if < 0.02

3. **L5 Avg badge** — new badge showing `l5_avg` vs `line`:
   - `"L5: {l5_avg} | Line: {line}"` — instantly shows the gap
   - This is the key info the user wants to see

4. **EV sort** now sorts by `model_prob` (true model edge), distinct from hit_rate sort.

---

### Step 6 — Handle std_dev Properly for EV Calculation
**File:** `utils/props_cache.py`

Instead of using a fixed `std_val = 4.0` in the live odds enrichment, precompute a per-player std before the enrichment loop:

```python
# Precompute player stats for enrichment (before the live odds loop)
_player_stats_cache: dict = {}
for prop in props_data:
    player = prop["player"]
    stat   = prop["stat"]
    key    = (player, stat)
    if key not in _player_stats_cache:
        p_df_tmp = DF[DF["PLAYER_NAME"] == player]
        if stat in p_df_tmp.columns and not p_df_tmp.empty:
            vals = pd.to_numeric(p_df_tmp[stat], errors="coerce").dropna()
            _player_stats_cache[key] = {
                "std": float(vals.std()) if len(vals) > 1 else 4.0,
                "vals_l10": vals.head(10),
            }
        else:
            _player_stats_cache[key] = {"std": 4.0, "vals_l10": pd.Series(dtype=float)}
```

Then use `_player_stats_cache[(player, stat)]["std"]` in the EV calc.

---

## Key Files

| File | Lines | Operation | Description |
|------|-------|-----------|-------------|
| `utils/props_cache.py` | ~285-315 | Modify | Change line baseline from L10 median → L5 average |
| `utils/props_cache.py` | ~301-315 | Modify | Store `ml_pred_stored` and add to prop dict |
| `utils/props_cache.py` | ~391-410 | Modify | Add `l5_avg`, `model_pred` to prop return dict |
| `utils/props_cache.py` | ~342-410 | Modify | Pass `l5_avg` into `_make_prop` closure |
| `utils/props_cache.py` | ~498-524 | Replace | Full live-odds enrichment rewrite (hit_rate + EV) |
| `utils/props_cache.py` | ~525-536 | Add | Quality gate + dedup + hard cap |
| `dashboard/app.py` | ~2660-2760 | Modify | Show L5 avg badge, fix EV badge, edge display |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `calculate_hit_probability` import missing in props_cache.py | Already imported via prop_calculator; verify import at top of file |
| std_dev of 0 for new players | Fallback: `std_dev = max(prediction * 0.25, 3.0)` |
| Quality gate too strict → 0 props shown | Use `_MIN_MODEL_PROB = 0.58` not 0.65; keep lock threshold lower; don't hard-gate locks |
| L5 avg line too high → hit_rate drops below 52% | Fine — that's correct. These were false high-hit-rate props. EV will still qualify if model_prob > 58% |
| Performance of per-prop DF lookups in enrichment | Use precomputed `_player_stats_cache` dict (Step 6) |
| Combo stats don't have model_pred | Fall back to `l5_avg` of the combined stat (already stored as `avg` in combo props) |

---

## Expected Outcome

- Barrett showing "Over 16.5" (real sportsbook line) with "L5: 20.5 | Line: 16.5" and "+EV: 18%" instead of meaningless "Over 10.5 — 90% hit rate"
- Total props reduced from ~80+ to ~20-25 high-quality picks
- EV sort ≠ Hit Rate sort: EV ranks by model probability edge, Hit Rate ranks by historical frequency
- Props that only qualify due to artificially low model-generated lines are eliminated by the quality gate

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (ace-tool not available; plan generated by Claude directly)
- GEMINI_SESSION: N/A
