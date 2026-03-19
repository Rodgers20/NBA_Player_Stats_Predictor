---
id: nba-traded-player-lookup
trigger: "when looking up a player's current team or position"
confidence: 0.85
domain: data
source: local-repo-analysis
analyzed_commits: 52
---

# Always Refresh Roster Before Player Lookup

## Action
When looking up a traded player's team or position, always ensure the roster cache is fresh.
Do not rely on stale `player_positions.csv` for traded players.

## Evidence
- Commit 966cb5e: "Fix: Data update issue - ensure traded players are found by refreshing roster"
- `data/player_positions.csv` can be stale after trade deadline

## Pattern
```python
# Trigger roster refresh if player not found
player = find_player(player_id, positions_df)
if player is None:
    refresh_roster()
    positions_df = load_positions()
    player = find_player(player_id, positions_df)
```
