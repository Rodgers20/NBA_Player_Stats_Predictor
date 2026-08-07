# WNBA Data Source Decision

**Date:** 2026-08-03
**Phase:** WNBA Phase 1

## Chosen Source: `nba_api` with `league_id="10"`

The `nba_api` Python package (already a project dependency) fully supports WNBA
data via its standard endpoints when passed `league_id="10"` (the WNBA league
identifier; NBA is `"00"`).

### Why not Kaggle?

Surveyed `natoshakennebrew/wnba-gamelogs-2015-2024`:
- Team-level game logs: 2015–2025, 4,512 rows across 10 seasons — usable but redundant.
- Player-level game logs: 2025 only, 5,407 rows — insufficient for multi-season training.

Also surveyed WNBA-related Kaggle datasets — none provide multi-season
player-level box scores comparable to `eoinamoore/historical-nba-data-and-player-box-scores`.

### Why `nba_api` wins

Empirical spike (see `/tmp/wnba_nba_api.py`):

```
WNBA 2025 player game logs: 5,407 rows in 0.7s
WNBA 2024 player game logs: 4,515 rows in 0.2s
WNBA 2023 player game logs: 4,544 rows in 0.1s
```

- Full 3-season backfill completes in **~1 second**.
- Always current (no dataset staleness).
- **Schema already matches our target format** — no translation layer needed
  (unlike the NBA path where we transform Kaggle columns → pipeline columns).

## Schema Map

`nba_api.stats.endpoints.LeagueGameLog(season, league_id="10", player_or_team_abbreviation="P")`
returns 32 columns that map 1:1 to our `EXPECTED_PLAYER_LOG_COLUMNS`
(defined in `utils/kaggle_loader.py:200`):

| nba_api column | Pipeline column | Notes |
|---|---|---|
| `SEASON_ID` | `SEASON_ID` | Format: `22025` for 2025 WNBA regular season |
| `PLAYER_ID` | `Player_ID` | Rename only |
| `PLAYER_NAME` | `PLAYER_NAME` | |
| `TEAM_ID` | (not in NBA schema) | New — WNBA needs it for logo URLs |
| `TEAM_ABBREVIATION` | (derived from MATCHUP for NBA) | |
| `GAME_ID` | `Game_ID` | Rename only |
| `GAME_DATE` | `GAME_DATE` | Format `"YYYY-MM-DD"` — needs reformat to `"Mmm dd, YYYY"` to match NBA pipeline |
| `MATCHUP` | `MATCHUP` | Already in `TOR vs. CHI` / `TOR @ CHI` format |
| `WL` | `WL` | "W"/"L" |
| `MIN` | `MIN` | |
| `FGM`, `FGA`, `FG_PCT` | same | |
| `FG3M`, `FG3A`, `FG3_PCT` | same | |
| `FTM`, `FTA`, `FT_PCT` | same | |
| `OREB`, `DREB`, `REB` | same | |
| `AST`, `STL`, `BLK`, `TOV`, `PF` | same | |
| `PTS`, `PLUS_MINUS` | same | |
| `VIDEO_AVAILABLE` | `VIDEO_AVAILABLE` | |
| `FANTASY_PTS` | (unused) | Drop |

**Derived columns** added downstream (matching NBA path):
- `SEASON` — string like `"2025"` (WNBA convention: single year, not `"2024-25"`)

## Team ID Range

WNBA team IDs live in `1611661313`–`1611661331` (nba_api convention).

**Active 2025 teams (13):**

| Abbrev | Team | ID |
|---|---|---|
| ATL | Atlanta Dream | 1611661330 |
| CHI | Chicago Sky | 1611661329 |
| CON | Connecticut Sun | 1611661323 |
| DAL | Dallas Wings | 1611661321 |
| GSV | Golden State Valkyries (2025 expansion) | 1611661331 |
| IND | Indiana Fever | 1611661325 |
| LAS | Los Angeles Sparks | 1611661320 |
| LVA | Las Vegas Aces | 1611661322 |
| MIN | Minnesota Lynx | 1611661324 |
| NYL | New York Liberty | 1611661313 |
| PHX | Phoenix Mercury | 1611661317 |
| SEA | Seattle Storm | 1611661328 |
| WAS | Washington Mystics | 1611661319 |

Team IDs will be pulled dynamically from `LeagueGameLog` output rather than
hardcoded — the mapping above is documentation only.

## Coverage

- **Historical:** `nba_api` returns full WNBA history via `season="YYYY"`
  (single year, not `"YYYY-YY"`). Confirmed working for 2015–2025.
- **Current:** 2025 season data is present through 2025-09-11.
- **Live/future:** `nba_api` returns games as they're recorded — no lag beyond
  official stats.wnba.com posting.

## Season Format Difference

- **NBA:** `"2024-25"` (spans two calendar years, Oct–Jun)
- **WNBA:** `"2025"` (single calendar year, May–Sep)

`utils/league_config.py` will expose season format helpers per league.

## Caching Strategy

WNBA data will be cached to `data/wnba/raw/{season}.csv` (parquet later once
we're sure of the schema). Refresh policy: on-demand via
`scripts/setup_data.py --league wnba --seasons 2023,2024,2025`.

No kagglehub cache for WNBA (unlike NBA which uses `KAGGLEHUB_CACHE=/Volumes/T7/kagglehub`).

## Dependencies

- `nba_api` — already installed in venv (`/Users/rodgersbahati/Developer/Projects/NBA/nba_env`).
- No new pip packages required for Phase 1.
