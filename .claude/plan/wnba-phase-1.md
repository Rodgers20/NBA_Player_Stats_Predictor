# WNBA Integration — Phase 1: Data + Models + Read-Only Player Analysis Page

## Scope

Ship a working `/wnba` player analysis page with:
- WNBA historical box scores loaded from a real data source
- Retrained PTS/AST/REB predictors on WNBA data
- League toggle in navbar (NBA ⇄ WNBA), URLs become `/wnba/*`
- Only the Player Analysis page live under `/wnba` — Games and Best Props pages hidden until Phase 2/3

Explicitly OUT of scope (Phases 2 & 3):
- WNBA today's schedule / live game predictor
- WNBA odds fetcher, props cache, parlay builder
- Injury feed for WNBA
- Self-grading, prediction tracker for WNBA

## Task Type
- [x] Fullstack — backend refactor + data pipeline + frontend page/routing

---

## Step 0 — Data Source Spike (timebox: 2 hours)

**Goal:** Pick a WNBA historical box scores source before touching code. Everything downstream depends on this.

**Candidates (in preference order):**
1. **Kaggle datasets** — search for WNBA player box scores 2020–present. Look for: game-level granularity (`personId`, `gameId`, `points`, `assists`, `rebounds`, `minutes`, `home`, `opponentTeamId`), 3+ seasons of coverage, updated within last 60 days.
2. **`nba_api` WNBA endpoints** — `nba_api.stats.endpoints.leaguegamelog` accepts `league_id="10"` (WNBA). Backfill script iterates seasons, caches to CSV. Slower initial load (~30 min) but always current.
3. **`basketball-reference.com` scraper** — last resort. Rate-limited, brittle. Only if 1 & 2 both fail.

**Deliverable of step 0:** short doc `docs/wnba-data-source.md` with:
- Chosen source (name, URL, license/attribution)
- Schema map: source columns → our expected columns (parallel to `EXPECTED_PLAYER_LOG_COLUMNS` in `utils/kaggle_loader.py:200`)
- Coverage summary: seasons present, last game date, row counts
- If Kaggle: dataset slug (e.g., `owner/dataset-name`) for `WNBA_KAGGLE_DATASET` constant

**Kill criteria:** if no source gives ≥ 2 recent WNBA seasons with per-game box scores, stop and escalate — Phase 1 can't proceed.

---

## Step 1 — LEAGUE_CONFIG Registry + Directory Reorg

**Goal:** Introduce league as a first-class concept without breaking NBA behavior.

### 1a. Create `utils/league_config.py` (new file)

```python
# Pseudocode structure
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

League = Literal["nba", "wnba"]

@dataclass(frozen=True)
class LeagueConfig:
    key: League
    display_name: str                    # "NBA" / "WNBA"
    brand: str                           # "NBA Props AI" / "WNBA Props AI"
    kaggle_dataset: str                  # slug or "" if not kaggle-sourced
    data_source_type: Literal["kaggle", "nba_api", "csv"]
    team_id_range: Tuple[int, int]       # (min, max) team IDs
    team_abbrevs: dict[str, int]         # abbrev -> teamId
    logo_cdn_template: str               # URL template with {team_id}
    headshot_cdn_template: str           # URL template with {player_id}
    odds_api_sport: str                  # "basketball_nba" / "basketball_wnba"
    season_start_month: int              # 10 for NBA, 5 for WNBA
    data_dir: Path                       # data/nba/ or data/wnba/
    models_dir: Path                     # models/nba/ or models/wnba/

NBA = LeagueConfig(key="nba", ...)
WNBA = LeagueConfig(key="wnba", ...)

CONFIGS: dict[League, LeagueConfig] = {"nba": NBA, "wnba": WNBA}

def get_config(league: League) -> LeagueConfig:
    return CONFIGS[league]
```

### 1b. Move existing files to `nba/` subdirs (git mv, don't rename in place)

```
data/*.csv                  → data/nba/*.csv
data/*.parquet              → data/nba/*.parquet
data/*_history.json         → data/nba/*_history.json
data/.last_kaggle_download  → data/nba/.last_kaggle_download
data/model_calibration.json → data/nba/model_calibration.json
models/*_predictor.pkl      → models/nba/*_predictor.pkl
models/game_*.pkl           → models/nba/game_*.pkl
```

Create empty `data/wnba/`, `models/wnba/`.

### 1c. Update path constants in loaders

Every module that references `data/` or `models/` gets updated to use `LeagueConfig.data_dir` / `models_dir`.

**Files affected (all `Modify`):**
- `utils/kaggle_loader.py` — `PROJECT_DATA_DIR` becomes `_project_root() / "data" / league`
- `models/predictor.py` — model path becomes `models/{league}/{stat}_predictor.pkl`
- `utils/prediction_tracker.py` — history file paths
- `utils/parlay_tracker.py` — history file paths
- `dashboard/app.py` — every `data/…` and `models/…` reference

### 1d. Rename `NBAPredictor` → `StatPredictor`

`models/predictor.py`: class rename + all import sites. No behavioral change.

**Success check:** existing NBA dashboard still boots and serves HTTP 200 after step 1. No regression.

---

## Step 2 — WNBA Data Loader

**Goal:** Mirror `utils/kaggle_loader.py` for WNBA, using whatever source step 0 selected.

### 2a. Refactor `utils/kaggle_loader.py` to take `league` parameter

Every public function gains `league: League = "nba"` param:
- `download_dataset(force=False, league="nba")`
- `_get_dataset_path(league="nba")`
- `load_player_game_logs(num_seasons=3, league="nba")`
- `load_team_stats(num_seasons=3, league="nba")`
- `load_team_defensive_stats(num_seasons=3, league="nba")`
- `load_player_positions(num_seasons=3, league="nba")`
- `export_pipeline_csvs(num_seasons=3, league="nba")`

Internally: dataset slug, team ID range, `PROJECT_DATA_DIR` all come from `get_config(league)`.

`KAGGLE_DATASET`, `_NBA_TEAM_ID_MIN`, `_NBA_TEAM_ID_MAX` constants deleted (they live in `LeagueConfig` now).

Rename module → `utils/stats_loader.py` (since it's no longer NBA-only nor Kaggle-only). Update all imports.

### 2b. Add non-Kaggle data path (if step 0 chose nba_api)

New `utils/nba_api_loader.py`:
- `_backfill_from_nba_api(seasons, league)` → downloads via `nba_api.stats.endpoints.leaguegamelog` with `league_id="10"` for WNBA
- Caches raw CSV in `data/wnba/raw/` (parallel to kagglehub cache structure)
- `stats_loader._get_dataset_path("wnba")` reads from this cache

### 2c. Team abbreviation registry for WNBA

WNBA has ~12 teams. Build the abbrev → teamId map by hand in `LEAGUE_CONFIG` (small, stable). Example:
```
ATL (Dream), CHI (Sky), CON (Sun), DAL (Wings),
IND (Fever), LVA (Aces), LAS (Sparks), MIN (Lynx),
NYL (Liberty), PHX (Mercury), SEA (Storm), WAS (Mystics),
GSV (Valkyries - 2025 expansion)
```

Team IDs: WNBA uses IDs in the `1611661313`–`1611661330` range (nba_api convention). Confirm during step 0 spike.

### 2d. Export WNBA CSVs

Run `python scripts/setup_kaggle_data.py --league wnba` (add `--league` flag to that script). Produces:
- `data/wnba/player_game_logs.csv`
- `data/wnba/team_stats.csv`
- `data/wnba/team_defensive_stats.csv`
- `data/wnba/player_positions.csv`
- `data/wnba/defense_vs_position.csv`
- `data/wnba/engineered_data.parquet`

**Success check:** `python -c "from utils.stats_loader import load_player_game_logs; print(load_player_game_logs(num_seasons=1, league='wnba').shape)"` returns > 5000 rows.

---

## Step 3 — Feature Engineering (league-aware)

**Goal:** `utils/feature_engineering.py` works for both leagues.

**Changes:**
- `add_rolling_averages(df, league="nba")` — add param, currently no NBA-specific hardcoding but add for symmetry
- Position inference thresholds may differ (WNBA guards avg more assists than NBA guards) — expose thresholds via `LeagueConfig`:
  ```
  # in LeagueConfig
  position_thresholds: {"G_min_ast": 4.0, "C_min_reb": 7.0}  # NBA
  position_thresholds: {"G_min_ast": 3.5, "C_min_reb": 6.5}  # WNBA (initial guess, tune later)
  ```
- Update `stats_loader.load_player_positions` to read thresholds from config

**Success check:** existing NBA rolling averages unchanged (byte-identical parquet); WNBA parquet builds without error.

---

## Step 4 — Model Training

**Goal:** Retrain PTS/AST/REB predictors on WNBA data, save to `models/wnba/`.

### 4a. Refactor `scripts/train_improved_models.py`

Add `--league {nba,wnba}` flag (default `nba` for backward compat).

Training loop uses `models/{league}/{stat}_predictor.pkl` as output path.

### 4b. Refactor `models/predictor.py::StatPredictor`

- Constructor takes `league` param
- Feature list may differ slightly (e.g., WNBA doesn't have exactly the same team pace stats — start with feature parity, tune later)
- `save()` and `load(stat, league)` handle the new path structure

### 4c. Train WNBA models

```
python scripts/train_improved_models.py --league wnba --stat PTS
python scripts/train_improved_models.py --league wnba --stat AST
python scripts/train_improved_models.py --league wnba --stat REB
```

Success criteria (initial baselines, refine in Phase 2):
- PTS MAE ≤ 4.5 (NBA baseline was ~5.5)
- AST MAE ≤ 1.8
- REB MAE ≤ 2.5

If MAE is significantly worse than NBA baselines, log it — Phase 1 still ships with the model; Phase 2 can iterate.

### 4d. Model calibration

Skip for Phase 1. `data/wnba/model_calibration.json` will just be `{}`. Calibration accumulates over Phase 3 once we're grading predictions.

---

## Step 5 — Dashboard: League Toggle + `/wnba` Player Analysis Page

**Goal:** Users can toggle NBA↔WNBA in navbar and see a working WNBA player analysis page.

### 5a. URL routing changes

Current router in `dashboard/app.py` (via `dcc.Location` callback at line ~2900 range — verify):
- `/` → NBA player analysis
- `/games` → NBA games
- `/props` → NBA best props

New router:
- `/` → redirect to `/nba/`
- `/nba/` → NBA player analysis  (existing, unchanged behavior)
- `/nba/games` → NBA games
- `/nba/props` → NBA best props
- `/wnba/` → WNBA player analysis (NEW)
- `/wnba/games` → "Coming soon" placeholder page
- `/wnba/props` → "Coming soon" placeholder page

### 5b. Navbar league toggle

`dashboard/app.py` navbar (~line 704):
- Add pill toggle: `[NBA] [WNBA]` styled, right side of navbar
- Toggle updates `dcc.Location` pathname to swap `/nba/*` ↔ `/wnba/*`
- Active league highlighted, brand title changes ("NBA Props AI" / "WNBA Props AI")

### 5c. Refactor page-creation functions to accept `league`

- `create_player_analysis_page(league)` — parametrize:
  - Data source globals (`PLAYER_DF`, `PLAYER_POSITIONS`, `TEAM_STATS`, `TEAM_DEFENSIVE_STATS`) become dict-of-dicts keyed by league: `PLAYER_DF["nba"]`, `PLAYER_DF["wnba"]`
  - Team ID → logo URL uses `LeagueConfig.logo_cdn_template`
  - Player headshot URL uses `LeagueConfig.headshot_cdn_template` (fallback to placeholder if WNBA CDN URL scheme unknown)
- 37 existing callbacks: audit each. Most probably access globals via league key.

### 5d. Data load at startup

`dashboard/app.py` startup loads BOTH leagues' data:
```
PLAYER_DF = {"nba": _load_league_data("nba"), "wnba": _load_league_data("wnba")}
```

Memory cost: WNBA is much smaller than NBA (12 teams, ~4-month season, ~150 players vs 30 teams, ~7-month season, ~600 players). Roughly +15% RAM. Acceptable.

### 5e. Header/branding

- Nav brand title reads from `get_config(current_league).brand`
- Current league tracked via URL prefix; helper: `_league_from_pathname(pathname) -> League`

**Success check:**
1. `/` redirects to `/nba/`, existing NBA experience unchanged
2. `/wnba/` loads player analysis page with WNBA players (search a known player like "A'ja Wilson")
3. Toggling league in navbar swaps the URL and content
4. `/wnba/games` and `/wnba/props` show "Coming in Phase 2/3" placeholder, not error
5. HTTP 200 on all routes

---

## Step 6 — Tests

**Goal:** Prevent regressions in NBA + basic coverage for WNBA path.

### 6a. Update existing tests
- `tests/test_data_fetch.py` — mock-based tests already work, add `league="nba"` explicit arg where signatures changed
- `tests/test_player_stats.py` — same

### 6b. Add WNBA parallel tests
- `tests/test_wnba_data_fetch.py` — same pattern as NBA test, using WNBA team IDs and fixture CSVs
- `tests/test_league_config.py` — verify both configs load, paths exist, key fields present
- `tests/test_dashboard_routing.py` (new) — smoke test that Dash app registers `/nba/` and `/wnba/` routes

Target coverage: 60%+ on new code, don't regress existing NBA coverage.

### 6c. Run full suite
```
pytest tests/ -v
```
All tests green before merging.

---

## Step 7 — Verification Checklist (Before Marking Phase 1 Done)

- [ ] `data/nba/*` and `data/wnba/*` both populated
- [ ] `models/nba/*_predictor.pkl` and `models/wnba/*_predictor.pkl` both present
- [ ] `python -c "import dashboard.app"` succeeds
- [ ] Dashboard starts on port 8050, HTTP 200 on `/`, `/nba/`, `/wnba/`
- [ ] Existing NBA player analysis works identically (spot-check 3 players, compare charts before/after)
- [ ] WNBA player analysis loads for a known WNBA player (A'ja Wilson, Caitlin Clark, Napheesa Collier)
- [ ] Navbar toggle switches leagues, URL changes correctly
- [ ] Placeholders present at `/wnba/games` and `/wnba/props` (not errors)
- [ ] `pytest tests/` all green
- [ ] `git diff` reviewed — no accidental data files or `._` sidecars committed (add `._*` to `.gitignore` if not already)
- [ ] `docs/wnba-data-source.md` committed
- [ ] README updated with WNBA setup steps (`--league wnba` flags for scripts)

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `utils/league_config.py` | Create | LeagueConfig dataclass + NBA/WNBA registry |
| `utils/stats_loader.py` | Rename+Modify | Renamed from `kaggle_loader.py`, gains `league` param |
| `utils/nba_api_loader.py` | Create (if step 0 → nba_api) | WNBA backfill via nba_api |
| `utils/feature_engineering.py` | Modify | Accept `league` param, thresholds from config |
| `utils/data_fetch.py` | Modify | League-aware nba_api / roster helpers |
| `utils/prediction_tracker.py` | Modify | League-aware history file paths |
| `utils/parlay_tracker.py` | Modify | League-aware history file paths |
| `utils/props_cache.py` | Modify (minimal) | Guard against WNBA calls in Phase 1 |
| `models/predictor.py` | Modify | Rename class to `StatPredictor`, add `league` param |
| `models/nba/*.pkl` | Move | From `models/*.pkl` |
| `data/nba/*` | Move | From `data/*` |
| `data/wnba/*` | Create | Via setup script |
| `models/wnba/*.pkl` | Create | Via training script |
| `scripts/setup_kaggle_data.py` | Modify | Add `--league` flag; rename to `setup_data.py` |
| `scripts/train_improved_models.py` | Modify | Add `--league` flag |
| `dashboard/app.py` | Modify | League toggle, `/wnba/*` routes, globals become dicts-per-league, page functions parametrized |
| `dashboard/assets/custom.css` | Modify (small) | Toggle pill styles |
| `tests/test_data_fetch.py` | Modify | Add `league="nba"` args |
| `tests/test_player_stats.py` | Modify | Same |
| `tests/test_wnba_data_fetch.py` | Create | WNBA parallel |
| `tests/test_league_config.py` | Create | Config sanity checks |
| `tests/test_dashboard_routing.py` | Create | Route registration smoke test |
| `docs/wnba-data-source.md` | Create | Data source decision record |
| `README.md` | Modify | Setup steps for `--league wnba` |
| `.gitignore` | Modify | Ensure `._*` and per-league data files excluded appropriately |

Approximate count: **~25 files touched**, 3 significant creates, 1 rename, 8 file moves.

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Step 0 finds no viable WNBA data source | Fall back to `nba_api` backfill (slower but always works). Kill Phase 1 only if `nba_api` WNBA endpoints are unreliable — very unlikely. |
| Refactor breaks NBA dashboard | After each substep, run existing tests + boot dashboard + spot-check known NBA player. Revert last commit if broken. |
| WNBA models perform badly (high MAE) | Ship Phase 1 anyway with a `⚠️ preliminary` badge on WNBA predictions. Phase 2/3 iterations improve them. Don't block Phase 1 on model quality. |
| Player headshot / team logo CDN missing for WNBA | Use `https://cdn.wnba.com/headshots/wnba/latest/1040x760/{player_id}.png` if it exists; else fallback to generic silhouette PNG in `dashboard/assets/`. |
| 6500-line `dashboard/app.py` gets messier | This refactor is opinionated: page functions parametrized, globals bucketed by league. Resist temptation to fully rewrite in Phase 1 — that's a separate cleanup PR. |
| Team abbrev collision (ATL exists in both leagues) | League-scoped: `TEAM_ABBREVS["nba"]["ATL"]` vs `TEAM_ABBREVS["wnba"]["ATL"]`. No collision because scoping is explicit. |
| Season timing (currently mid-WNBA season) means live data expectations rise | Set expectations: Phase 1 is stats browser only. `/wnba/games` and `/wnba/props` explicitly say "Coming in Phase 2/3". |
| ExFAT sidecar files on T7 (Kaggle cache) | Already using external T7 for `KAGGLEHUB_CACHE`; if WNBA data goes there too, same treatment. Venv stays on APFS internal (already done). |
| Dashboard load time grows | WNBA data is ~1/4 NBA size. Impact is small. If it becomes visible, load WNBA lazily on first `/wnba/*` request. |

---

## Estimated Effort

- Step 0 (data source spike): 2 hours
- Step 1 (config + refactor + moves): 4 hours
- Step 2 (WNBA loader): 3–6 hours (depends on source)
- Step 3 (feature engineering): 1 hour
- Step 4 (model training): 2 hours (mostly waiting)
- Step 5 (dashboard routing + toggle): 6–8 hours (touches most of app.py)
- Step 6 (tests): 3 hours
- Step 7 (verification): 1 hour

**Total: 22–27 hours of focused work.** Realistically 3–4 working days.

---

## SESSION_ID (for /ccg:execute use)

- CODEX_SESSION: N/A (codeagent-wrapper not installed locally; plan authored by Claude directly)
- GEMINI_SESSION: N/A (same reason)
