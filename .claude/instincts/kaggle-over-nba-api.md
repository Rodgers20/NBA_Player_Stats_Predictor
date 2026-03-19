---
id: nba-kaggle-over-api
trigger: "when fetching NBA historical data or player game logs"
confidence: 0.95
domain: data
source: local-repo-analysis
analyzed_commits: 52
---

# Use Kaggle Dataset, Not NBA API, for Historical Data

## Action
Always use `utils/kaggle_loader.py` for historical player/game data.
Only use `nba_api` for live roster lookups or today's schedule — and wrap in try/except with fallback.

## Evidence
- Commit d2ab2dc: "feat: replace NBA API with Kaggle dataset for historical data"
- Commit 48f81c8: "fix: remove NBA API dependency from Best Props, H2H, and Today's Games"
- Commit 68a4832: "fix: harden NBA API calls and add smart daily Kaggle auto-update"
- NBA API is fragile for historical queries; Kaggle provides stable, complete data

## Pattern
```python
# CORRECT: Use kaggle_loader for historical
from utils.kaggle_loader import load_player_logs
logs = load_player_logs(player_id)

# RISKY: Live NBA API (wrap with fallback)
try:
    roster = nba_api.get_today_roster()
except Exception:
    roster = cached_roster
```
