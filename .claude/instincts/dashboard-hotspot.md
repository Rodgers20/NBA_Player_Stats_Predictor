---
id: nba-dashboard-hotspot
trigger: "when adding a new dashboard feature or UI component"
confidence: 0.95
domain: architecture
source: local-repo-analysis
analyzed_commits: 52
---

# dashboard/app.py Is the Single Source of Truth

## Action
When implementing any dashboard feature, modify `dashboard/app.py` for layout and callbacks.
Keep business logic in `utils/` — never embed data fetching or ML inference directly in callbacks.

## Evidence
- `dashboard/app.py` changed in 30+ of 52 commits (most changed file by far)
- Pattern: feature commits always touch app.py + one or more utils/
- CSS-only changes go to `dashboard/assets/custom.css`

## Co-change Pattern
- Dashboard feature → `dashboard/app.py` + `utils/<feature_util>.py`
- Styling fix → `dashboard/app.py` + `dashboard/assets/custom.css`
- Data change → `utils/data_fetch.py` + `utils/data_updater.py`
