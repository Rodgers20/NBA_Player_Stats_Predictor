---
name: nba-player-stats-predictor-patterns
description: Coding patterns extracted from NBA_Player_Stats_Predictor git history
version: 1.0.0
source: local-git-analysis
analyzed_commits: 52
generated: 2026-03-07
---

# NBA Player Stats Predictor — Patterns

## Commit Conventions

This project uses a **mixed style** — newer commits adopt conventional commits, older commits use imperative titles:

- `feat:` — New features (dashboard tabs, data sources, ML models)
- `fix:` — Bug fixes (often followed by specific component: "Fix Best Props to...")
- `perf:` — Performance improvements (caching, background pre-computation)
- Older style: `Fix <Component> to <do X>`, `Add <Feature>`, `Update <thing>`

**Recommended going forward:** Use conventional commits consistently:
```
feat: add new stat category to Best Props
fix: correct player lookup for traded players
perf: cache defensive stats to avoid re-fetching
```

## Code Architecture

```
NBA_Player_Stats_Predictor/
├── dashboard/
│   ├── app.py          # ← PRIMARY hotspot (Plotly Dash app, callbacks, layout)
│   └── assets/
│       └── custom.css  # Dashboard styling
├── utils/
│   ├── data_fetch.py       # NBA data retrieval (most-changed utility)
│   ├── data_updater.py     # Background auto-update scheduler
│   ├── kaggle_loader.py    # Kaggle dataset ingestion
│   ├── feature_engineering.py  # ML feature transforms
│   ├── prop_calculator.py  # Prop bet EV calculations
│   ├── prop_scorer.py      # Prop scoring/ranking
│   ├── props_cache.py      # Pre-computed props cache
│   ├── injury_news.py      # Injury/news scraping
│   └── teammate_impact.py  # Teammate context features
├── models/
│   └── predictor.py        # XGBoost/sklearn model training & inference
├── scripts/                # One-off data collection scripts
├── data/                   # CSV + Parquet data files
├── tests/                  # pytest test files
├── notebooks/              # Jupyter exploration notebooks
├── Dockerfile              # HuggingFace Spaces deployment
├── render.yaml             # Render.com deployment config
├── conftest.py             # Adds project root to sys.path
└── requirements.txt
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dashboard | Plotly Dash |
| ML Models | XGBoost, scikit-learn |
| Data | Pandas, Parquet (pyarrow), Kaggle |
| Scheduling | APScheduler (background tasks) |
| Testing | pytest |
| Deployment | Render (gunicorn), Hugging Face Spaces (Docker) |
| Data Source | Kaggle dataset (replaced NBA API) |

## Key Workflows

### Adding a Dashboard Feature
1. Add/modify callbacks in `dashboard/app.py`
2. Add supporting logic in `utils/<relevant_util>.py`
3. Style in `dashboard/assets/custom.css`
4. Test via `tests/test_player_stats.py` or `tests/test_data_fetch.py`

### Updating the Data Pipeline
1. Modify `utils/kaggle_loader.py` for ingestion changes
2. Update `utils/data_fetch.py` for fetch/roster logic
3. Update `utils/data_updater.py` for scheduling changes
4. Re-run relevant `scripts/collect_*.py` for local refresh

### Adding/Retraining ML Models
1. Update `utils/feature_engineering.py` for new features
2. Retrain via `models/predictor.py` or a notebook
3. Verify output Parquet: `data/engineered_data.parquet`
4. Update `utils/prop_calculator.py` if EV logic changes

### Performance Optimization
Pattern: pre-compute expensive results on startup/background, cache to `utils/props_cache.py`:
- APScheduler triggers background refresh
- Parquet used for fast DataFrame caching
- Lazy model loading on first request

### Deployment
- **Render**: `render.yaml` → `gunicorn dashboard.app:server`
- **HuggingFace**: `Dockerfile` → containerized Dash app
- Python version: 3.10.12

## Testing Patterns

- Framework: **pytest**
- Test files: `tests/test_*.py`
- Root `conftest.py` adds project root to `sys.path`
- Coverage: aim for 80%+ on `utils/` and `models/`
- Key test files:
  - `tests/test_data_fetch.py` — data retrieval logic
  - `tests/test_player_stats.py` — player stat computations

## Data Files (Do Not Edit Manually)

| File | Purpose |
|------|---------|
| `data/player_game_logs.csv` | Raw per-game logs |
| `data/player_positions.csv` | Current roster positions |
| `data/defense_vs_position.csv` | Defensive matchup data |
| `data/team_defensive_stats.csv` | Team defense metrics |
| `data/team_stats.csv` | Team aggregate stats |
| `data/engineered_data.parquet` | ML-ready feature matrix |

## Common Pitfalls (from git history)

1. **Traded players** — always refresh roster before lookup (see commit 966cb5e)
2. **NBA API fragility** — prefer Kaggle dataset over live NBA API for historical data
3. **Tooltip CSS conflicts** — Dash slider tooltips need component-level `tooltip=None` plus CSS override
4. **DataFrame iteration** — use `.iterrows()` or vectorized ops, not raw iteration (see commit 62af4e5)
5. **Startup perf** — expensive computations (Best Props) must run in background thread, not on page load
