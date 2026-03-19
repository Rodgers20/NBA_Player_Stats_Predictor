---
id: nba-background-precompute
trigger: "when adding expensive computation triggered on page load"
confidence: 0.9
domain: performance
source: local-repo-analysis
analyzed_commits: 52
---

# Pre-Compute Expensive Results in Background, Never on Page Load

## Action
Use APScheduler or a background thread to pre-compute expensive results (e.g. Best Props scoring).
Cache results in `utils/props_cache.py`. Callbacks read from cache, not compute inline.

## Evidence
- Commit a1b9065: "perf: pre-compute Best Props in background for instant page loads"
- Commit 7598ee9: "Optimize startup: Parquet caching + lazy model loading"
- Parquet used for fast DataFrame persistence between runs

## Pattern
```python
# WRONG: compute in callback
@app.callback(...)
def update_best_props():
    return compute_all_props()  # slow

# CORRECT: read from cache
@app.callback(...)
def update_best_props():
    return props_cache.get_cached()  # fast

# Background refresh (APScheduler)
scheduler.add_job(refresh_props_cache, 'interval', hours=1)
```
