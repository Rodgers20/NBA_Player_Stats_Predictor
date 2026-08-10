# WNBA Synthetic-Lines Fallback

## Goal
When The Odds API is unreachable (quota exhausted, network down, etc.), still generate WNBA props by synthesizing a line from the player's own recent form (L20 average). Model projection stays the same; only the line source changes.

## Design
- **Synthetic line** = round(L20 average, nearest 0.5). For example, Nneka L20 REB avg = 9.5 → line = 9.5.
- **Odds assumed at -110/-110** (standard vig) when no book is available.
- **Pick decision**: model projection vs synthetic line — same OVER/UNDER logic as before.
- **UI marker**: synthetic props tagged with "L20 line" instead of a bookmaker name so users know it's a self-generated projection, not a live sportsbook line.
- **Priority**: use real sportsbook odds when available; only synthesize for players/stats not covered by live odds. Merged, not either/or.

## Implementation
1. **`utils/wnba_props.py`**
   - New helper `_synthetic_line_from_recent(recent_df, stat)` — returns half-point-rounded L20 average.
   - Extend `generate_wnba_props` with a `synthesize_missing: bool = True` flag. After processing real odds, for every (player, stat) combo not already in the props list, synthesize a line and generate a prop with `bookmaker="L20 avg"` and `over_price = under_price = -110`.
2. **`dashboard/app.py::_wnba_props_row`**
   - Show "L20 line" chip (different color) when `bookmaker == "L20 avg"` so it's visually distinct.
3. **Empty-state message on `/wnba/props`**
   - No longer says "no props" when synthetic props are available. Instead shows "no sportsbook odds — synthesizing from L20 form" as a subtitle.
4. **Tests**
   - Synthetic line rounds correctly.
   - Empty odds dict + synthesize_missing=True → props generated for all eligible players.
   - Real odds + synthesize_missing=True → real odds win; synthetic only fills gaps.

## Files
| File | Op |
|---|---|
| `utils/wnba_props.py` | Modify (synthetic helper + generator flag) |
| `dashboard/app.py` | Modify (row renderer marks synthetic; props page copy) |
| `tests/test_wnba_props.py` | Add 3 synthetic tests |

## Commit + Push
- Commit locally with a descriptive message
- Push to GitHub `origin/main`
- Push to Hugging Face remote (if configured)
