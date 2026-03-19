# Implementation Plan: Best Props + Data Pipeline Fix

## Root Cause Diagnosis

Before any fix: **the Kaggle download is broken.**

```
~/.cache/kagglehub/datasets/eoinamoore/.../versions/392/  →  EMPTY DIRECTORY
```

`kaggle_loader.py` silently falls through to `data/engineered_data.parquet` (Oct 31, 2025).
All 4 issues trace back to this single root cause:

| Issue | Root Cause |
|-------|-----------|
| Data 4 months stale (Oct 31, 2025) | Kaggle download silently fails → stale parquet |
| Best Props shows no games / empty | No current-season stats → no valid props for today's teams |
| NBA API today's games failing | `scoreboardv2` is fragile; no robust fallback |
| Injury data wrong | Nitter RSS mirrors go down; no reliable fallback |

---

## Fix 1 — Repair the Kaggle Download Pipeline

**Files**: `utils/kaggle_loader.py`, `utils/data_updater.py`, `scripts/setup_kaggle_data.py`

### 1a. Add visible error reporting to `download_dataset()`

```python
def download_dataset(force: bool = False) -> Path:
    import kagglehub, os
    # Log auth status before attempting download
    username = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_CONFIG_DIR")
    logger.info(f"Kaggle auth: username={username or 'NOT SET'}")

    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET, force_download=force)
        result_path = Path(path)
        # Verify the download actually has files
        csv = result_path / "PlayerStatistics.csv"
        if not csv.exists():
            raise FileNotFoundError(f"Downloaded directory is empty: {result_path}")
        size_mb = csv.stat().st_size / 1e6
        logger.info(f"Download OK: {result_path} ({size_mb:.1f} MB)")
        return result_path
    except Exception as e:
        logger.error(f"Kaggle download FAILED: {e}")
        raise
```

### 1b. Add startup forced refresh in `data_updater.py`

Move the daily check to also fire on startup (don't wait 30 min for first update):

```python
# In update_game_data(), change daily check to:
should_force = (_last_download_date != today) or (_last_download_date is None)
if should_force:
    try:
        download_dataset(force=True)
        _last_download_date = today
    except Exception as e:
        logger.warning(f"Force download failed, trying cached: {e}")
        download_dataset(force=False)  # Use whatever is cached
```

### 1c. Add a `scripts/force_refresh_data.py` one-shot fix script

```python
#!/usr/bin/env python3
"""Run this once to force-pull latest Kaggle data into data/ CSVs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.kaggle_loader import download_dataset, export_pipeline_csvs
import kagglehub

print("Step 1: Checking Kaggle credentials...")
# kagglehub uses ~/.config/kaggle/kaggle.json OR env vars KAGGLE_USERNAME + KAGGLE_KEY
# If neither is set, it will fail with a clear message.

print("Step 2: Force-downloading latest dataset...")
download_dataset(force=True)

print("Step 3: Exporting to data/ CSVs + parquet...")
export_pipeline_csvs(num_seasons=3)

print("Done. Check data/engineered_data.parquet for latest dates.")
```

### 1d. Fix `export_pipeline_csvs` to also write parquet

After exporting CSVs, also write `engineered_data.parquet` so app startup is fast:

```python
# At end of export_pipeline_csvs():
from utils.feature_engineering import add_rolling_averages
print("Adding rolling averages and writing parquet...")
enriched = add_rolling_averages(game_logs)
enriched["_date"] = pd.to_datetime(enriched["GAME_DATE"], format="%b %d, %Y", errors="coerce")
enriched.to_parquet(PROJECT_DATA_DIR / "engineered_data.parquet", index=False)
print(f"  -> parquet written ({len(enriched)} rows)")
```

---

## Fix 2 — Robust Today's Games (Multi-Source Fallback)

**File**: `utils/data_fetch.py`

The current `get_todays_games()` uses NBA API only. Replace with a 3-source cascade:

```
Source 1: NBA API scoreboardv2  (fast, authoritative when working)
    ↓ fails
Source 2: ESPN hidden JSON API  (reliable, free, no auth)
    ↓ fails
Source 3: Kaggle DF - filter today's games  (works offline, uses existing data)
    ↓ no games found
Return empty (it's genuinely an off day)
```

### ESPN API call (no auth required):
```
GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
Returns JSON with today's games including teams, status, scores
```

### Implementation:

```python
def _get_games_from_espn() -> pd.DataFrame:
    """Fallback: ESPN's hidden JSON scoreboard API (no auth)."""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    try:
        resp = requests.get(url, headers=CUSTOM_HEADERS, timeout=10)
        data = resp.json()
        rows = []
        for event in data.get("events", []):
            comp = event["competitions"][0]
            teams = {t["homeAway"]: t["team"]["abbreviation"] for t in comp["competitors"]}
            rows.append({
                "GAME_ID": event["id"],
                "GAME_DATE_EST": event["date"][:10],
                "HOME_TEAM": teams.get("home", ""),
                "AWAY_TEAM": teams.get("away", ""),
                "GAME_STATUS_TEXT": event["status"]["type"]["description"],
                "HOME_TEAM_ID": 0,
                "VISITOR_TEAM_ID": 0,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        logger.warning(f"ESPN games fallback failed: {e}")
        return pd.DataFrame()

def get_todays_games() -> pd.DataFrame:
    """Fetch today's games with 3-source cascade fallback."""
    # ... existing cache check ...

    # Source 1: NBA API
    result = _get_games_from_nba_api(today)
    if not result.empty:
        _cache_and_return(result)
        return result

    # Source 2: ESPN (no auth)
    logger.info("NBA API failed, trying ESPN...")
    result = _get_games_from_espn()
    if not result.empty:
        _cache_and_return(result)
        return result

    # Source 3: Kaggle DF (look for today's games in historical data)
    logger.info("ESPN failed, checking Kaggle data for today's schedule...")
    result = _get_games_from_df_cache(today)
    _cache_and_return(result)
    return result
```

---

## Fix 3 — Contextual Player Insight Cards

**New file**: `utils/insight_generator.py`
**Modified**: `utils/props_cache.py` (add `insight` field to each prop dict)

### Insight structure:

```python
{
    "narrative": "Averaging 26.4 PPG in last 5 games (hot streak), facing the Hornets who rank 28th vs guards. Expect an over-performance.",
    "trend": "hot",        # hot | cold | neutral
    "blowout_risk": False,
    "blowout_label": "",
    "rest_days": 2,
    "matchup_grade": "A",  # A/B/C/D based on def rank
    "factors": [
        {"text": "28th-ranked defense vs guards", "positive": True},
        {"text": "5+ point scorer in 9 of last 10", "positive": True},
        {"text": "2 days rest", "positive": True},
        {"text": "Away game (averaging 2.1 fewer pts away)", "positive": False},
    ]
}
```

### Key logic:

```python
def generate_player_insight(player_name, stat, line, opponent, player_df,
                             defense_vs_pos, is_home, position):
    insight = {}
    recent_5 = player_df.head(5)[stat]
    recent_10 = player_df.head(10)[stat]
    season_avg = player_df[player_df["SEASON"] == CURRENT_SEASON][stat].mean()

    # Trend: hot if last-5 > last-10 avg by >10%
    l5_avg = recent_5.mean()
    l10_avg = recent_10.mean()
    trend_pct = (l5_avg - l10_avg) / (l10_avg + 0.001)
    if trend_pct > 0.10:
        trend = "hot"
    elif trend_pct < -0.10:
        trend = "cold"
    else:
        trend = "neutral"

    # Defensive context
    opp_rank = get_def_rank(opponent, position, stat, defense_vs_pos)
    # rank 1-10 = strong defense, 21-30 = weak defense
    matchup_grade = "A" if opp_rank >= 24 else "B" if opp_rank >= 18 else "C" if opp_rank >= 10 else "D"

    # Blowout risk: if opponent W% > 70% and player's team W% < 30%
    blowout_risk = check_blowout_risk(player_df, opponent)

    # Home/away split
    home_avg = player_df[player_df["MATCHUP"].str.contains("vs.", na=False)].head(10)[stat].mean()
    away_avg = player_df[player_df["MATCHUP"].str.contains("@", na=False)].head(10)[stat].mean()
    location_delta = home_avg - away_avg  # positive = better at home

    # Rest days
    if len(player_df) >= 2:
        last_game_date = player_df["_date"].iloc[0]
        rest_days = (datetime.now().date() - last_game_date.date()).days
    else:
        rest_days = 1

    # Build narrative
    trend_word = {"hot": "🔥 on a hot streak", "cold": "struggling recently", "neutral": "averaging"}[trend]
    insight["narrative"] = build_narrative(
        player_name, stat, l5_avg, trend_word, opp_rank, matchup_grade,
        blowout_risk, is_home, location_delta, rest_days)
    insight["trend"] = trend
    insight["blowout_risk"] = blowout_risk
    insight["rest_days"] = rest_days
    insight["matchup_grade"] = matchup_grade
    insight["factors"] = build_factors(
        opp_rank, trend, blowout_risk, is_home, location_delta, rest_days, l5_avg, line)

    return insight


def build_narrative(player, stat, l5_avg, trend_word, opp_rank, grade,
                    blowout, home, delta, rest):
    stat_label = {"PTS": "points", "AST": "assists", "REB": "rebounds", "FG3M": "threes"}[stat]
    loc_ctx = "at home" if home else "on the road"
    def_ctx = f"#{opp_rank} defense (weakest in the league vs this position)" if opp_rank >= 26 \
              else f"#{opp_rank} defense vs this position"
    blowout_ctx = " Blowout risk — stars may sit 4th quarter." if blowout else ""
    delta_ctx = f" Averages {abs(delta):.1f} {'more' if (home and delta>0) or (not home and delta<0) else 'fewer'} {stat_label} {loc_ctx}." if abs(delta) > 1 else ""

    return (f"{trend_word.capitalize()} {loc_ctx}, averaging {l5_avg:.1f} {stat_label} over last 5. "
            f"Facing {def_ctx}.{delta_ctx}{blowout_ctx}")
```

### Blowout risk logic:

```python
def check_blowout_risk(player_df, opponent):
    """
    Blowout risk: when a team is heavily favored by 15+ points,
    star players often sit the 4th quarter. Check if team/opponent
    win percentage differential is extreme.
    We use recent PLUS_MINUS as proxy — if player's team has been
    outscored by 10+ per game recently, likely to blow out.
    """
    if "PLUS_MINUS" not in player_df.columns:
        return False
    recent_pm = player_df.head(10)["PLUS_MINUS"].mean()
    # If team consistently loses by big margins, opponent will likely win big
    # When losing big, stars play fewer minutes
    return recent_pm < -10
```

---

## Fix 4 — Reliable Injury Data

**File**: `utils/injury_news.py`

Replace the unreliable Nitter/RSS sources with a 3-source cascade:

### Source Priority:
1. **ESPN Injury API** (hidden, free, no auth) — authoritative
2. **RotoWire scrape** (existing, keep as backup)
3. **NBA official injury report PDF** (released daily by NBA league)

### ESPN Injury API:
```
GET https://site.api.espn.com/apis/fantasy/v2/games/fba/games
    ?useMap=true&dates={YYYYMMDD}&pbpOnly=true
Returns injuries listed as "questionable", "out", "day-to-day"
```

```python
def _fetch_espn_injury_data() -> dict:
    """
    Returns {player_name: {"status": "OUT"|"QUESTIONABLE"|"PROBABLE", "reason": str}}
    """
    url = f"https://site.api.espn.com/apis/fantasy/v2/games/fba/games"
    today = datetime.now().strftime("%Y%m%d")
    resp = requests.get(url, params={"useMap": "true", "dates": today}, timeout=10)
    data = resp.json()
    result = {}
    for team in data.get("players", {}).values():
        for player in team:
            name = player.get("displayName", "")
            injury = player.get("injury", {})
            if injury:
                status = injury.get("status", "ACTIVE").upper()
                desc = injury.get("description", "")
                result[name] = {"status": status, "reason": desc}
    return result
```

Add a 4-hour cache (injury status doesn't change minute-to-minute):

```python
_injury_cache = {"data": {}, "timestamp": None}
INJURY_CACHE_TTL = 4 * 3600  # 4 hours

def get_all_injury_data() -> dict:
    """Fetch all injury data with 4-hour TTL cache."""
    now = datetime.now()
    if (_injury_cache["timestamp"] and
        (now - _injury_cache["timestamp"]).total_seconds() < INJURY_CACHE_TTL):
        return _injury_cache["data"]

    data = {}
    # Source 1: ESPN
    try:
        data.update(_fetch_espn_injury_data())
    except Exception as e:
        logger.warning(f"ESPN injuries failed: {e}")

    # Source 2: RotoWire (existing scraper)
    if not data:
        try:
            data.update(_fetch_rotowire_injuries())
        except Exception as e:
            logger.warning(f"RotoWire injuries failed: {e}")

    _injury_cache["data"] = data
    _injury_cache["timestamp"] = now
    return data
```

---

## Implementation Steps

### Step 1: Diagnose and fix Kaggle credentials (IMMEDIATE)
1. Check if `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars are set (or set up `~/.config/kaggle/kaggle.json`)
2. Run `python scripts/force_refresh_data.py` to force-pull all data
3. Verify `data/engineered_data.parquet` now has dates through March 2026

### Step 2: Harden `kaggle_loader.py`
- `download_dataset()`: add file existence check + size validation
- Add verbose logging so failures are visible, not silent
- Persist `_last_download_date` to a file so it survives restarts

### Step 3: Add ESPN fallback to `get_todays_games()`
- Refactor into 3 private methods (`_from_nba_api`, `_from_espn`, `_from_df_cache`)
- Public `get_todays_games()` cascades through all 3
- Separate the NBA API logic for easier testing

### Step 4: Add `utils/insight_generator.py`
- `generate_player_insight(player_name, stat, line, ...)` → insight dict
- `build_narrative(...)` → human-readable string
- `check_blowout_risk(...)` → bool
- Call from `_compute_main_page_props()` in `props_cache.py`
- Add `insight` key to each prop dict

### Step 5: Fix injury data in `injury_news.py`
- Add `_fetch_espn_injury_data()` as primary source
- Add 4-hour cache for all injury data
- Remove Nitter sources (unreliable)
- Keep RotoWire as fallback

### Step 6: Trigger startup refresh
- In `app.py`, call `scheduled_update()` immediately at startup (not just in 30 min)
- This ensures fresh data is loaded before the props cache warms

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `utils/kaggle_loader.py:40-52` | Modify | Add download validation + verbose error |
| `utils/data_updater.py:111-117` | Modify | Persist `_last_download_date`, startup trigger |
| `utils/data_fetch.py:238-297` | Modify | Add ESPN + DF fallback to `get_todays_games()` |
| `utils/insight_generator.py` | Create | New file: narrative insight engine |
| `utils/props_cache.py:110-184` | Modify | Call `generate_player_insight()`, add to prop dict |
| `utils/injury_news.py:80-87` | Modify | ESPN primary source + 4hr cache |
| `scripts/force_refresh_data.py` | Create | One-shot data refresh script |
| `dashboard/app.py:262-265` | Modify | Call `scheduled_update()` at startup |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|-----------|
| Kaggle auth not configured → download fails | `force_refresh_data.py` prints clear instructions; app falls back to CSV |
| ESPN API changes endpoint → today's games fails | NBA API is still tried first; DF fallback still works |
| Insight narrative is wrong for a player | Insight is additive (shown in addition to existing stats); won't break props display |
| ESPN injury API doesn't have all players | RotoWire scraper kept as fallback |
| Blowout risk false positives | Conservative threshold (avg PLUS_MINUS < -10) |
